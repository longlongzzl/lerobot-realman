#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import mat2quat

from rm75_app.assets.object_specs import OBJECT_SPECS, get_object_spec, normalize_object_name, resolve_object_spec_scales
from rm75_app.placement.place_rules import DESK_SLOT_LAYOUT_XZ, get_place_rule
from rm75_app.paths import APP_ROOT, DEFAULT_CUROBO_CFG, RUNTIME_DIR


os.environ.setdefault("MPLCONFIGDIR", "/tmp/rm75_matplotlib")
DEFAULT_DIRECT_SCRIPT = "rm75_app.runtime.curobo2_pick_place"
DEFAULT_OUTPUT_ROOT = RUNTIME_DIR / "llm_pick_place_runs"
LLM_PLAN_SCHEMA_VERSION = "rm75_pick_place_plan_v1"
DEFAULT_ROBOT_BASE_WORLD_XYZ = np.asarray([-0.615, 0.0, 0.0], dtype=np.float32)
DEFAULT_LLM_PROVIDER = os.environ.get("RM75_LLM_PROVIDER", "mock")
DEFAULT_LLM_MODEL = os.environ.get("RM75_LLM_MODEL", "")
DEFAULT_LLM_API_BASE = os.environ.get("RM75_LLM_API_BASE", "")
DEFAULT_LLM_API_KEY_ENV = os.environ.get("RM75_LLM_API_KEY_ENV", "")


def _table_direction_xy(direction: str) -> np.ndarray:
    """Canonical tabletop language frame: +X is front, +Y is left."""
    key = str(direction or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "左": "left",
        "右": "right",
        "前": "front",
        "后": "back",
        "左前": "left_front",
        "前左": "left_front",
        "右前": "right_front",
        "前右": "right_front",
        "左后": "left_back",
        "后左": "left_back",
        "右后": "right_back",
        "后右": "right_back",
        "front_left": "left_front",
        "front_right": "right_front",
        "back_left": "left_back",
        "back_right": "right_back",
    }
    key = aliases.get(key, key)
    direction_vecs = {
        "front": np.array([1.0, 0.0], dtype=np.float32),
        "back": np.array([-1.0, 0.0], dtype=np.float32),
        "left": np.array([0.0, 1.0], dtype=np.float32),
        "right": np.array([0.0, -1.0], dtype=np.float32),
        "left_front": np.array([1.0, 1.0], dtype=np.float32),
        "right_front": np.array([1.0, -1.0], dtype=np.float32),
        "left_back": np.array([-1.0, 1.0], dtype=np.float32),
        "right_back": np.array([-1.0, -1.0], dtype=np.float32),
    }
    vec = direction_vecs.get(key, np.array([0.0, -1.0], dtype=np.float32))
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 1e-6 else vec


CN_OBJECT_ALIASES: dict[str, list[str]] = {
    "lvmukuai": ["绿木块", "绿色木块", "绿色方块", "绿方块", "木块", "积木"],
    "carriot": ["胡萝卜", "萝卜", "carrot"],
    "shuazi": ["刷子", "毛刷"],
    "hongshupian": ["薯片盒", "薯片桶", "红薯片", "薯片罐", "薯片"],
    "gluestick": ["胶棒", "胶水", "固体胶", "胶棒瓶"],
    "bi": ["红笔", "笔", "这支笔"],
    "tennis": ["网球", "球", "黄球", "tennis"],
    "bitong": ["笔筒", "笔杯", "杯筒", "高笔筒", "矮笔筒"],
    "desk": ["桌子", "桌面", "桌", "台面"],
}

AMBIGUITY_WORDS = {
    "tall": ["高的", "高一点", "较高", "高笔筒", "高"],
    "short": ["矮的", "低的", "较矮", "矮笔筒", "矮"],
    "left": ["左边", "左侧", "左"],
    "right": ["右边", "右侧", "右"],
    "front": ["前面", "前边", "前"],
    "back": ["后面", "后边", "后"],
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _normalize_text(text: str) -> str:
    return str(text or "").strip().lower().replace(" ", "").replace("-", "_")


def _axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9 or abs(float(angle_rad)) <= 1e-12:
        return np.eye(3, dtype=np.float32)
    axis = axis / norm
    x, y, z = [float(v) for v in axis]
    c = float(math.cos(angle_rad))
    s = float(math.sin(angle_rad))
    C = 1.0 - c
    return np.asarray(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float32,
    )


def _matrix_to_rpy_deg(T: np.ndarray) -> tuple[float, float, float]:
    rpy = mat2euler(np.asarray(T, dtype=np.float64).reshape(4, 4)[:3, :3], axes="sxyz")
    return tuple(float(math.degrees(v)) for v in rpy)


def _local_pose_from_matrix(T_local: np.ndarray) -> dict[str, Any]:
    T_local = np.asarray(T_local, dtype=np.float32).reshape(4, 4)
    return {
        "position": [float(v) for v in T_local[:3, 3]],
        "rpy_deg": [float(v) for v in _matrix_to_rpy_deg(T_local)],
    }


_MESH_BOUNDS_CACHE: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
_MESH_VERTICES_CACHE: dict[tuple[str, float], np.ndarray] = {}


def _mesh_bounds_scaled(spec_name: str) -> tuple[np.ndarray, np.ndarray]:
    spec = get_object_spec(spec_name)
    if spec is None:
        raise ValueError(f"Unknown object spec: {spec_name}")
    sim_asset = str(Path(spec.sim_asset_file or spec.mesh_file).expanduser())
    _, sim_scale = resolve_object_spec_scales(spec)
    key = (str(Path(sim_asset).resolve()), float(sim_scale))
    cached = _MESH_BOUNDS_CACHE.get(key)
    if cached is not None:
        return cached
    loaded = trimesh.load(sim_asset, force="scene")
    bounds = np.asarray(loaded.bounds, dtype=np.float32)
    if bounds.shape != (2, 3):
        raise RuntimeError(f"No mesh bounds available for {spec_name}: {sim_asset}")
    scaled = (bounds * float(sim_scale)).astype(np.float32)
    _MESH_BOUNDS_CACHE[key] = (scaled[0], scaled[1])
    return scaled[0], scaled[1]


def _mesh_vertices_scaled(spec_name: str) -> np.ndarray:
    spec = get_object_spec(spec_name)
    if spec is None:
        raise ValueError(f"Unknown object spec: {spec_name}")
    sim_asset = str(Path(spec.sim_asset_file or spec.mesh_file).expanduser())
    _, sim_scale = resolve_object_spec_scales(spec)
    key = (str(Path(sim_asset).resolve()), float(sim_scale))
    cached = _MESH_VERTICES_CACHE.get(key)
    if cached is not None:
        return cached
    loaded = trimesh.load(sim_asset, force="scene")
    if hasattr(loaded, "to_geometry"):
        mesh = loaded.to_geometry()
    elif hasattr(loaded, "dump"):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    vertices = (np.asarray(mesh.vertices, dtype=np.float32) * float(sim_scale)).astype(np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise RuntimeError(f"No mesh vertices available for {spec_name}: {sim_asset}")
    _MESH_VERTICES_CACHE[key] = vertices
    return vertices


def _bbox_corners(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    mins = np.asarray(mins, dtype=np.float32).reshape(3)
    maxs = np.asarray(maxs, dtype=np.float32).reshape(3)
    return np.asarray(
        [
            [x, y, z]
            for x in (mins[0], maxs[0])
            for y in (mins[1], maxs[1])
            for z in (mins[2], maxs[2])
        ],
        dtype=np.float32,
    )


def _world_points(T_world_obj: np.ndarray, spec_name: str) -> np.ndarray:
    mins, maxs = _mesh_bounds_scaled(spec_name)
    pts = _bbox_corners(mins, maxs)
    T = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def _world_z_bounds(T_world_obj: np.ndarray, spec_name: str) -> tuple[float, float]:
    pts = _world_points(T_world_obj, spec_name)
    return float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))


def _precise_world_z_bounds(T_world_obj: np.ndarray, spec_name: str) -> tuple[float, float]:
    vertices = _mesh_vertices_scaled(spec_name)
    T = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
    z = (T[:3, :3] @ vertices.T).T[:, 2] + float(T[2, 3])
    return float(np.min(z)), float(np.max(z))


def _precise_projected_xy_extents(T_world_obj: np.ndarray, spec_name: str) -> np.ndarray:
    vertices = _mesh_vertices_scaled(spec_name)
    T = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
    xy = (T[:3, :3] @ vertices.T).T[:, :2] + T[:2, 3]
    return (np.max(xy, axis=0) - np.min(xy, axis=0)).astype(np.float32)


def _footprint_radius(spec_name: str) -> float:
    mins, maxs = _mesh_bounds_scaled(spec_name)
    ext = np.asarray(maxs - mins, dtype=np.float32)
    return float(max(ext[0], ext[1], ext[2]) * 0.5)


def _adjust_bottom_to_z(T_world_obj: np.ndarray, spec_name: str, z_bottom: float, margin: float = 0.002) -> np.ndarray:
    T = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4).copy()
    current_bottom, _ = _world_z_bounds(T, spec_name)
    T[2, 3] += float(z_bottom) + float(margin) - float(current_bottom)
    return T


def _infer_spec_name(object_id: str, entry: dict[str, Any]) -> str | None:
    for raw in (entry.get("object_name"), entry.get("spec_name"), object_id):
        normalized = normalize_object_name(raw)
        if normalized in OBJECT_SPECS:
            return normalized
        if normalized:
            for spec_name in OBJECT_SPECS:
                if normalized.startswith(f"{spec_name}_") or normalized.endswith(f"_{spec_name}"):
                    return spec_name
    return None


@dataclass
class SceneObject:
    object_id: str
    spec_name: str
    label: str
    score: float
    placed: bool
    T_world_obj: np.ndarray

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.T_world_obj[:3, 3], dtype=np.float32).reshape(3)

    def context_record(self) -> dict[str, Any]:
        spec = get_object_spec(self.spec_name)
        mins, maxs = _mesh_bounds_scaled(self.spec_name)
        zmin, zmax = _world_z_bounds(self.T_world_obj, self.spec_name)
        prompt = "" if spec is None else str(spec.grounding_prompt)
        return {
            "object_id": self.object_id,
            "spec_name": self.spec_name,
            "label": self.label,
            "grounding_prompt": prompt,
            "aliases": CN_OBJECT_ALIASES.get(self.spec_name, []),
            "position_xyz_m": [round(float(v), 5) for v in self.position],
            "world_z_min_m": round(float(zmin), 5),
            "world_z_max_m": round(float(zmax), 5),
            "asset_extent_xyz_m": [round(float(v), 5) for v in (maxs - mins)],
            "score": float(self.score),
            "placed": bool(self.placed),
            "T_world_obj": np.asarray(self.T_world_obj, dtype=np.float32).tolist(),
        }


class SceneState:
    def __init__(self, scene_file: Path, objects: dict[str, SceneObject], raw_data: dict[str, Any]):
        self.scene_file = scene_file
        self.objects = objects
        self.raw_data = raw_data

    @classmethod
    def load(cls, path: str | Path) -> "SceneState":
        scene_file = Path(path).expanduser().resolve()
        data = json.loads(scene_file.read_text(encoding="utf-8"))
        raw_objects = data.get("objects", {})
        if not isinstance(raw_objects, dict):
            raise ValueError(f"Scene file {scene_file} has no top-level objects dictionary")
        objects: dict[str, SceneObject] = {}
        for object_id, entry in raw_objects.items():
            if not isinstance(entry, dict) or "T_world_obj" not in entry:
                continue
            spec_name = _infer_spec_name(object_id, entry)
            if spec_name is None:
                continue
            objects[str(object_id)] = SceneObject(
                object_id=str(object_id),
                spec_name=str(spec_name),
                label=str(entry.get("label", object_id)),
                score=float(entry.get("score", 1.0)),
                placed=bool(entry.get("placed", False)),
                T_world_obj=np.asarray(entry["T_world_obj"], dtype=np.float32).reshape(4, 4),
            )
        if not objects:
            raise ValueError(f"Scene file {scene_file} has no usable objects")
        return cls(scene_file, objects, data)

    def copy(self) -> "SceneState":
        return SceneState(
            self.scene_file,
            {
                key: SceneObject(
                    object_id=obj.object_id,
                    spec_name=obj.spec_name,
                    label=obj.label,
                    score=obj.score,
                    placed=obj.placed,
                    T_world_obj=obj.T_world_obj.copy(),
                )
                for key, obj in self.objects.items()
            },
            json.loads(json.dumps(self.raw_data, ensure_ascii=False)),
        )

    def get(self, object_id: str) -> SceneObject:
        if object_id not in self.objects:
            raise KeyError(f"Object {object_id!r} is not in the scene")
        return self.objects[object_id]

    def find_by_spec(self, spec_name: str) -> list[SceneObject]:
        normalized = normalize_object_name(spec_name)
        return [obj for obj in self.objects.values() if obj.spec_name == normalized]

    def update_pose(self, object_id: str, T_world_obj: np.ndarray, *, placed: bool = True) -> None:
        obj = self.get(object_id)
        obj.T_world_obj = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4).copy()
        obj.placed = bool(placed)

    def context(self) -> dict[str, Any]:
        assets = []
        for spec_name, spec in sorted(OBJECT_SPECS.items()):
            try:
                mins, maxs = _mesh_bounds_scaled(spec_name)
                extent = [round(float(v), 5) for v in (maxs - mins)]
            except Exception:
                extent = None
            assets.append(
                {
                    "spec_name": spec_name,
                    "grounding_prompt": spec.grounding_prompt,
                    "aliases": CN_OBJECT_ALIASES.get(spec_name, []),
                    "real_longest_axis_m": spec.real_longest_axis_m,
                    "asset_extent_xyz_m": extent,
                    "grasp_mode": spec.grasp_mode,
                }
            )
        slots = []
        if "desk" in self.objects:
            desk = self.objects["desk"]
            for slot_name, _ in DESK_SLOT_LAYOUT_XZ:
                try:
                    T_slot, _ = _slot_pose(self, desk, slot_name)
                    slots.append(
                        {
                            "slot": slot_name,
                            "position_xyz_m": [round(float(v), 5) for v in T_slot[:3, 3]],
                            "note": "small_desk slot world position; use directly when user says N号位",
                        }
                    )
                except Exception:
                    pass
        return {
            "source_scene_file": str(self.scene_file),
            "coordinate_frame": {
                "world_units": "meters",
                "front_axis": "+X",
                "back_axis": "-X",
                "left_axis": "+Y",
                "right_axis": "-Y",
                "up_axis": "+Z",
                "instruction": "For complex spatial commands, compute the target position from these object/slot coordinates and output explicit coordinates.",
            },
            "object_count": len(self.objects),
            "objects": [obj.context_record() for obj in sorted(self.objects.values(), key=lambda item: item.object_id)],
            "small_desk_slots": slots,
            "assets": assets,
            "supported_goal_operators": [
                "inside(target)",
                "between(object_a, object_b)",
                "pose_goal(position_expr, orientation_expr)",
                "on(target)",
                "relative(target, direction, distance_m)",
                "upright_in_place()",
                "rotate_in_place(yaw_deg)",
                "exchange(object_a, object_b, temporary_buffer_pose)",
            ],
        }

    def write_fixed_scene(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(json.dumps(self.raw_data, ensure_ascii=False))
        raw_objects = data.setdefault("objects", {})
        for object_id, obj in self.objects.items():
            entry = raw_objects.setdefault(object_id, {})
            entry["T_world_obj"] = np.asarray(obj.T_world_obj, dtype=np.float32).tolist()
            entry["label"] = obj.label
            entry["score"] = float(obj.score)
            entry["placed"] = bool(obj.placed)
            entry["spec_name"] = obj.spec_name
        data["source"] = str(data.get("source", "llm_orchestrator"))
        data["llm_orchestrator_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def write_command_fixed_scene(self, path: str | Path, *, source_id: str, target_id: str) -> tuple[Path, list[str]]:
        """Write a direct-script scene view for the selected step.

        The semantic layer can distinguish instance IDs such as bitong_tall and
        bitong_short. The current low-level executor still selects active objects
        by canonical object spec, so the chosen source/target are materialized
        under their canonical spec names for this one command. Extra duplicate
        instances of the same spec are omitted from this command view.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(json.dumps(self.raw_data, ensure_ascii=False))
        raw_objects: dict[str, Any] = {}
        warnings: list[str] = []
        important_ids = {str(source_id), str(target_id)}
        used_keys: set[str] = set()

        def _entry_for(obj: SceneObject) -> dict[str, Any]:
            return {
                "T_world_obj": np.asarray(obj.T_world_obj, dtype=np.float32).tolist(),
                "label": obj.label,
                "score": float(obj.score),
                "placed": bool(obj.placed),
                "spec_name": obj.spec_name,
            }

        ordered_objects: list[SceneObject] = []
        for object_id in (source_id, target_id):
            if object_id in self.objects and self.objects[object_id] not in ordered_objects:
                ordered_objects.append(self.objects[object_id])
        ordered_objects.extend(obj for obj in self.objects.values() if obj not in ordered_objects)

        for obj in ordered_objects:
            if obj.object_id in important_ids:
                key = obj.spec_name
            else:
                key = obj.object_id if normalize_object_name(obj.object_id) in OBJECT_SPECS else obj.spec_name
            if key in used_keys:
                warnings.append(
                    f"omitted duplicate instance {obj.object_id!r} for executor key {key!r}; "
                    "LLM context still keeps the full instance list"
                )
                continue
            used_keys.add(key)
            raw_objects[key] = _entry_for(obj)
        data["objects"] = raw_objects
        data["source"] = str(data.get("source", "llm_orchestrator_command_view"))
        data["llm_orchestrator_command_view"] = {
            "source_id": source_id,
            "target_id": target_id,
            "note": "selected source/target are keyed by canonical object spec for the current executor",
        }
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return out, warnings


def _object_tokens(obj: SceneObject) -> list[str]:
    tokens = {obj.object_id, obj.spec_name, obj.label}
    tokens.update(CN_OBJECT_ALIASES.get(obj.spec_name, []))
    spec = get_object_spec(obj.spec_name)
    if spec is not None:
        tokens.add(spec.grounding_prompt)
    return sorted({_normalize_text(t) for t in tokens if str(t).strip()}, key=len, reverse=True)


def _adjective_bias(text: str) -> str | None:
    normalized = _normalize_text(text)
    for key, words in AMBIGUITY_WORDS.items():
        if any(_normalize_text(word) in normalized for word in words):
            return key
    return None


def _oriented_height(obj: SceneObject) -> float:
    zmin, zmax = _world_z_bounds(obj.T_world_obj, obj.spec_name)
    return float(zmax - zmin)


def resolve_object_ref(scene: SceneState, text: str, *, role: str) -> dict[str, Any]:
    normalized = _normalize_text(text)
    scored: list[tuple[float, SceneObject, str]] = []
    for obj in scene.objects.values():
        best_score = 0.0
        best_token = ""
        for token in _object_tokens(obj):
            if token and token in normalized:
                score = float(len(token))
                if obj.object_id == token or obj.spec_name == token:
                    score += 2.0
                if score > best_score:
                    best_score = score
                    best_token = token
        if best_score > 0.0:
            scored.append((best_score, obj, best_token))
    if not scored:
        raise ValueError(f"Could not resolve {role} object from text: {text!r}")
    scored.sort(key=lambda item: (-item[0], item[1].object_id))
    top_score = scored[0][0]
    tied = [item for item in scored if abs(item[0] - top_score) <= 1e-6]
    bias = _adjective_bias(text)
    if len(tied) > 1 and bias in {"tall", "short"}:
        reverse = bias == "tall"
        tied.sort(key=lambda item: _oriented_height(item[1]), reverse=reverse)
    elif len(tied) > 1 and bias in {"left", "right", "front", "back"}:
        if bias == "left":
            tied.sort(key=lambda item: float(item[1].position[0]))
        elif bias == "right":
            tied.sort(key=lambda item: -float(item[1].position[0]))
        elif bias == "front":
            tied.sort(key=lambda item: -float(item[1].position[1]))
        elif bias == "back":
            tied.sort(key=lambda item: float(item[1].position[1]))
    chosen = tied[0][1]
    return {
        "text": text,
        "role": role,
        "object_id": chosen.object_id,
        "spec_name": chosen.spec_name,
        "matched_token": tied[0][2],
        "candidate_count": len(tied),
        "disambiguation": bias,
    }


def _find_mentions(scene: SceneState, text: str) -> list[dict[str, Any]]:
    normalized = _normalize_text(text)
    mentions = []
    for obj in scene.objects.values():
        for token in _object_tokens(obj):
            if not token:
                continue
            start = normalized.find(token)
            if start >= 0:
                mentions.append(
                    {
                        "start": start,
                        "end": start + len(token),
                        "object_id": obj.object_id,
                        "spec_name": obj.spec_name,
                        "token": token,
                    }
                )
                break
    mentions.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
    deduped = []
    seen = set()
    occupied_spans: list[tuple[int, int]] = []
    for item in mentions:
        if item["object_id"] in seen:
            continue
        if any(item["start"] < end and item["end"] > start for start, end in occupied_spans):
            continue
        seen.add(item["object_id"])
        occupied_spans.append((int(item["start"]), int(item["end"])))
        deduped.append(item)
    return deduped


def _split_command_sequence(command: str) -> list[str]:
    text = str(command or "").strip()
    if not text:
        return []
    parts = re.split(r"(?:然后|接着|之后|随后|并且|；|;|。|再把|再将|再)", text)
    return [part.strip() for part in parts if part and part.strip()]


def _orientation_policy_from_text(text: str) -> str | None:
    compact = _normalize_text(text)
    if any(word in compact for word in ("平放", "横放", "躺着放", "水平放")):
        return "horizontal"
    if any(word in compact for word in ("竖放", "立放", "竖着放", "垂直放")):
        return "vertical"
    return None


def _distance_m_from_text(text: str, default_m: float = 0.08) -> float:
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(厘米|公分|cm|CM|米|m)", str(text))
    if not match:
        return float(default_m)
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"厘米", "公分", "cm"}:
        return value / 100.0
    return value


def _has_explicit_distance_text(text: str) -> bool:
    return bool(re.search(r"[-+]?\d+(?:\.\d+)?\s*(厘米|公分|cm|CM|米|m)", str(text)))


def _fraction_from_text(text: str, default: float = 1.0 / 3.0) -> float:
    compact = _normalize_text(text)
    table = {
        "二分之一": 0.5,
        "一半": 0.5,
        "三分之一": 1.0 / 3.0,
        "三分之二": 2.0 / 3.0,
        "四分之一": 0.25,
        "四分之三": 0.75,
    }
    for token, value in table.items():
        if token in compact:
            return float(value)
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", str(text))
    if match:
        denom = float(match.group(2))
        if abs(denom) > 1e-9:
            return float(match.group(1)) / denom
    return float(default)


def _pose_goal_orientation_from_text(text: str, *, default_target_ref: str | None = None) -> dict[str, Any]:
    face_match = re.search(r"(?:尖头)?朝向(.+)$", str(text))
    if face_match:
        return {
            "type": "long_axis_toward",
            "target": face_match.group(1).strip(),
            "semantic_axis": "tip" if "尖头" in _normalize_text(text) else "long_axis",
        }
    if "桌边平行" in _normalize_text(text):
        return {"type": "parallel_table_edge"}
    if default_target_ref is not None:
        return {"type": "long_axis_toward", "target": default_target_ref, "semantic_axis": "long_axis"}
    return {"type": "keep_current"}


def _movable_scene_objects(scene: SceneState) -> list[SceneObject]:
    objects = [
        obj
        for obj in scene.objects.values()
        if obj.spec_name != "desk" and get_place_rule(obj.spec_name) is not None
    ]
    non_placed = [obj for obj in objects if not bool(obj.placed)]
    return non_placed or objects


def _attribute_object_token(scene: SceneState, text: str, *, role: str) -> str | None:
    compact = _normalize_text(text)
    candidates = _movable_scene_objects(scene)
    if "红色" in compact and any(word in compact for word in ("长条", "细长", "条状")):
        for preferred in ("bi", "carriot", "gluestick"):
            for obj in candidates:
                if obj.spec_name == preferred:
                    return obj.object_id
    if "黄色" in compact and any(word in compact for word in ("球", "圆球")):
        for obj in candidates:
            if obj.spec_name == "tennis":
                return obj.object_id
    if any(word in compact for word in ("长条", "细长", "条状")):
        long_items = []
        for obj in candidates:
            mins, maxs = _mesh_bounds_scaled(obj.spec_name)
            ext = np.asarray(maxs - mins, dtype=np.float32)
            aspect = float(np.max(ext) / max(np.min(ext), 1e-6))
            long_items.append((aspect, obj))
        long_items.sort(key=lambda item: (-item[0], item[1].object_id))
        if long_items:
            return long_items[0][1].object_id
    return None


def _mentions_small_desk(text: str) -> bool:
    compact = _normalize_text(text)
    return any(
        word in compact
        for word in ("小桌", "小桌子", "放置桌", "目标桌", "slot桌", "槽位桌", "号位桌")
    )


def _mock_llm_parse_single(command: str, scene: SceneState) -> dict[str, Any]:
    text = str(command).strip()
    compact = _normalize_text(text)
    mentions = _find_mentions(scene, text)
    orientation_policy = _orientation_policy_from_text(text)

    if "所有" in compact and "桌子" in compact and "1" in compact and "6" in compact and "号" in compact:
        slot_names = [f"slot_{idx}" for idx in range(1, 7)]
        objects = _collection_default_objects(
            scene,
            slots=True,
            order="left_to_right" if "从左到右" in compact else "",
        )
        return {
            "kind": "multi_step",
            "operator": "collection_slots",
            "source_refs": [obj.object_id for obj in objects[: len(slot_names)]],
            "slot_names": slot_names,
            "notes": "mock LLM chose collection left-to-right desk slots",
        }

    if any(word in compact for word in ("清理桌面", "整理桌面")) and "右半边" in compact:
        target_surface = "small_desk" if _mentions_small_desk(text) else "worktable"
        objects = _collection_default_objects(scene, slots=target_surface == "small_desk", order="")
        return {
            "kind": "multi_step",
            "operator": "collection_right_half",
            "source_refs": [obj.object_id for obj in objects],
            "avoid_overlap": "不要互相重叠" in compact,
            "target_surface": target_surface,
            "notes": f"mock LLM chose collection cleanup to {target_surface} right half",
        }

    exchange_like = (
        "交换" in compact
        or "互换" in compact
        or "调换" in compact
        or "换位置" in compact
        or "位置互换" in compact
        or ("位置" in compact and compact.startswith("交") and len(mentions) >= 2)
    )
    if exchange_like and len(mentions) >= 2:
        return {
            "kind": "multi_step",
            "operator": "exchange",
            "source_ref": mentions[0]["token"],
            "other_ref": mentions[1]["token"],
            "avoid_refs": [item["token"] for item in mentions[2:]] if "不要碰" in compact else [],
            "notes": "mock LLM chose exchange via keyword",
        }

    angle_match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*度", text)
    if ("旋转" in compact or "转" in compact) and angle_match and mentions:
        return {
            "kind": "single_step",
            "operator": "rotate_in_place",
            "source_ref": mentions[0]["token"],
            "yaw_deg": float(angle_match.group(1)),
            "notes": "mock LLM chose in-place yaw rotation",
        }

    if any(word in compact for word in ("立起来", "竖起来", "站起来", "扶正")) and mentions:
        return {
            "kind": "single_step",
            "operator": "upright_in_place",
            "source_ref": mentions[0]["token"],
            "notes": "mock LLM chose upright-in-place",
        }

    if any(word in compact for word in ("连线", "连线上")) and len(mentions) >= 3:
        line_a = mentions[1]
        line_b = mentions[2]
        near_ref = line_b
        near_match = re.search(r"靠近(.+?)(?:[一二三四五六七八九十\d./]+分之[一二三四五六七八九十\d]+|[一二三四五六七八九十\d./]+/[\d.]+|的位置|处|$)", text)
        if near_match:
            try:
                resolved_near = resolve_object_ref(scene, near_match.group(1), role="near_line_endpoint")
                for item in (line_a, line_b):
                    if item["object_id"] == resolved_near["object_id"]:
                        near_ref = item
                        break
            except Exception:
                near_ref = line_b
        fraction_near = float(np.clip(_fraction_from_text(text, 1.0 / 3.0), 0.0, 1.0))
        alpha = fraction_near if near_ref["object_id"] == line_a["object_id"] else 1.0 - fraction_near
        return {
            "kind": "single_step",
            "operator": "pose_goal",
            "source_ref": mentions[0]["token"],
            "pose_goal": {
                "position": {
                    "type": "linear_interpolation",
                    "from": line_a["token"],
                    "to": line_b["token"],
                    "alpha_from_to": alpha,
                    "surface": "worktable",
                },
                "orientation": _pose_goal_orientation_from_text(text, default_target_ref=near_ref["token"]),
            },
            "notes": "mock LLM chose generalized pose_goal on an object-object line",
        }

    if "中间" in compact and len(mentions) >= 3:
        return {
            "kind": "single_step",
            "operator": "pose_goal",
            "source_ref": mentions[0]["token"],
            "pose_goal": {
                "position": {
                    "type": "midpoint",
                    "object_a": mentions[1]["token"],
                    "object_b": mentions[2]["token"],
                    "surface": "worktable",
                },
                "orientation": _pose_goal_orientation_from_text(text),
            },
            "notes": "mock LLM chose generalized pose_goal midpoint",
        }

    if (
        any(word in compact for word in ("放进", "放入", "扔进", "丢进", "放进去", "里面", "内部"))
        and len(mentions) >= 2
    ):
        return {
            "kind": "single_step",
            "operator": "inside",
            "source_ref": mentions[0]["token"],
            "target_ref": mentions[1]["token"],
            "notes": "mock LLM chose inside/drop-into",
        }

    if (
        any(word in compact for word in ("靠在", "靠着", "靠到", "倚在", "倚靠", "依靠", "斜靠", "贴靠"))
        and len(mentions) >= 2
    ):
        side = "right"
        for direction, words in {
            "left_front": ["左前", "前左"],
            "right_front": ["右前", "前右"],
            "left_back": ["左后", "后左"],
            "right_back": ["右后", "后右"],
            "left": ["左边", "左侧", "左面"],
            "right": ["右边", "右侧", "右面"],
            "front": ["前面", "前边", "前侧"],
            "back": ["后面", "后边", "后侧"],
        }.items():
            if any(word in compact for word in words):
                side = direction
                break
        return {
            "kind": "single_step",
            "operator": "lean_against",
            "source_ref": mentions[0]["token"],
            "target_ref": mentions[1]["token"],
            "side": side,
            "lean_angle_deg": 32.0,
            "bottom_on": "table",
            "notes": "mock LLM chose lean-against placement",
        }

    if len(mentions) >= 1 and any(word in compact for word in ("位置", "空位", "地方")):
        for relation, words in {
            "left_front": ["左前", "前左"],
            "right_front": ["右前", "前右"],
            "left_back": ["左后", "后左"],
            "right_back": ["右后", "后右"],
            "frontmost": ["最前"],
            "backmost": ["最后", "最靠后"],
            "leftmost": ["最左"],
            "rightmost": ["最右"],
            "nearest": ["最近"],
            "farthest": ["最远"],
        }.items():
            if any(word in compact for word in words):
                return {
                    "kind": "single_step",
                    "operator": "extreme_empty",
                    "source_ref": mentions[0]["token"],
                    "target_ref": "desk",
                    "surface": "small_desk" if "小桌" in compact or "桌子" in compact else "worktable",
                    "relation": relation,
                    "reference": "robot_base",
                    "reference_frame": "table_xy",
                    "notes": "mock LLM chose table_xy extreme/corner empty placement",
                }

    for direction, words in {
        "left_front": ["左前", "前左"],
        "right_front": ["右前", "前右"],
        "left_back": ["左后", "后左"],
        "right_back": ["右后", "后右"],
        "left": ["左边", "左侧"],
        "right": ["右边", "右侧"],
        "front": ["前面", "前边"],
        "back": ["后面", "后边"],
    }.items():
        if any(word in compact for word in words) and len(mentions) >= 2:
            return {
                "kind": "single_step",
                "operator": "relative",
                "source_ref": mentions[0]["token"],
                "target_ref": mentions[1]["token"],
                "direction": direction,
                "distance_m": _distance_m_from_text(text, 0.08),
                "strict_distance": _has_explicit_distance_text(text),
                "parallel_to": "desk_edge" if "桌边平行" in compact else None,
                "notes": "mock LLM chose relative placement",
            }

    source_attr = _attribute_object_token(scene, text, role="source")
    target_attr = None
    if source_attr is not None and "后面" in compact:
        if "黄色" in compact and "球" in compact:
            target_attr = _attribute_object_token(scene, "黄色球", role="target")
        if target_attr is not None:
            return {
                "kind": "single_step",
                "operator": "relative",
                "source_ref": source_attr,
                "target_ref": target_attr,
                "direction": "back",
                "distance_m": _distance_m_from_text(text, 0.08),
                "strict_distance": _has_explicit_distance_text(text),
                "notes": "mock LLM resolved attribute-based relative placement",
            }

    if "最近" in compact and "空位" in compact and len(mentions) >= 2:
        return {
            "kind": "single_step",
            "operator": "nearest_empty",
            "source_ref": mentions[0]["token"],
            "target_ref": mentions[1]["token"],
            "surface": "small_desk" if _mentions_small_desk(text) else "worktable",
            "notes": "mock LLM chose nearest empty surface position",
        }

    slot_match = re.search(r"([1-6一二三四五六])\s*号", text)
    if slot_match and mentions:
        raw = slot_match.group(1)
        slot_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
        slot_num = int(slot_map.get(raw, raw))
        return {
            "kind": "single_step",
            "operator": "desk_slot",
            "source_ref": mentions[0]["token"],
            "slot_name": f"slot_{slot_num}",
            "notes": "mock LLM chose explicit desk slot",
        }

    if any(word in compact for word in ("叠到", "叠在")) and len(mentions) >= 2:
        return {
            "kind": "single_step",
            "operator": "on_or_side_fallback",
            "source_ref": mentions[0]["token"],
            "target_ref": mentions[1]["token"],
            "fallback_direction": "right",
            "notes": "mock LLM chose stack with stability fallback",
        }

    if any(word in compact for word in ("放到", "放在", "放上", "放")) and len(mentions) >= 2:
        return {
            "kind": "single_step",
            "operator": "on",
            "source_ref": mentions[0]["token"],
            "target_ref": mentions[1]["token"],
            "orientation_policy": orientation_policy,
            "anchor": "top_center" if any(word in compact for word in ("上面", "上边", "顶部", "顶上", "放上")) else "center",
            "notes": "mock LLM chose surface placement",
        }

    raise ValueError(f"Mock LLM could not parse command: {command!r}; mentions={mentions}")


def mock_llm_parse(command: str, scene: SceneState) -> dict[str, Any]:
    compact = _normalize_text(command)
    if "挡路" in compact and any(word in compact for word in ("放进", "扔进", "放入")):
        mentions = _find_mentions(scene, command)
        if len(mentions) >= 2:
            source_ref = mentions[0]["token"]
            target_ref = mentions[1]["token"]
            return {
                "kind": "single_step",
                "operator": "inside",
                "source_ref": source_ref,
                "target_ref": target_ref,
                "notes": (
                    "mock LLM skipped speculative blocker motion; the main placement should be "
                    "attempted first and blocker clearing should only be generated after a real failure"
                ),
            }
    parts = _split_command_sequence(command)
    if len(parts) > 1:
        return {
            "kind": "multi_step_sequence",
            "operator": "sequence",
            "steps": [_mock_llm_parse_single(part, scene) for part in parts],
            "notes": "mock LLM split command into sequential structured goals",
        }
    return _mock_llm_parse_single(command, scene)


def _llm_pick_place_interface_text() -> str:
    return f"""# RM75 Natural-Language Pick-Place LLM Interface

Output exactly one JSON object. Do not output markdown, comments, or extra text.

Schema version: {LLM_PLAN_SCHEMA_VERSION}

Top-level shape:

{{
  "schema_version": "{LLM_PLAN_SCHEMA_VERSION}",
  "user_command": "original user command",
  "assumptions": ["short assumptions, optional"],
  "steps": [
    {{
      "action": "pick_place",
      "object": "object reference from the scene or user wording",
      "goal": {{
        "type": "pose | inside | on_top | beside | lean_against | between | slot | nearest_empty | extreme_empty | upright_in_place | rotate_in_place",
        "...": "goal-specific fields"
      }}
    }}
  ]
}}

Object references can use either canonical IDs/specs or user-facing Chinese names from the scene context.
Prefer symbolic goal types when they match the user's language, because the local feasibility layer can add
clearance, choose a reachable surface point, and preserve the requested spatial relation. Use goal.type="pose"
only for exact coordinates, line ratios, custom orientation constraints, or geometry not covered by beside/between.
Coordinate convention for spatial language: +X is front/前, -X is back/后, +Y is left/左, -Y is right/右. Therefore right_front is (+X,-Y), left_front is (+X,+Y), right_back is (-X,-Y), left_back is (-X,+Y).
For relational spatial language, prefer symbolic operators such as beside, between, collection, nearest_empty,
extreme_empty, inside, on_top, slot, lean_against, or exchange.

Supported actions:

1. pick_place
   Required: object, goal.

2. exchange
   Required: object_a, object_b.
   Example:
   {{"action":"exchange","object_a":"绿木块","object_b":"网球"}}

3. collection
   Required: objects or "all_movable", goal.
   Example:
   {{"action":"collection","objects":"all_movable","goal":{{"type":"slots","surface":"small_desk","order":"left_to_right"}}}}

Goal types:

pose:
  {{"type":"pose",
    "position":{{"type":"linear_interpolation","from":"红薯片","to":"笔筒","alpha_from_to":0.667,"surface":"worktable"}},
    "orientation":{{"type":"long_axis_toward","target":"笔筒","semantic_axis":"tip"}}}}

  Supported position expressions:
  - {{"type":"absolute_xyz","xyz_m":[-0.30,0.15,0.04],"surface":"worktable","z_policy":"surface"}}
  - {{"type":"absolute_xy","xy_m":[-0.30,0.15],"surface":"small_desk"}}
  - {{"type":"object_center","object":"笔筒"}}
  - {{"type":"midpoint","object_a":"绿木块","object_b":"笔筒"}}
  - {{"type":"linear_interpolation","from":"红薯片","to":"笔筒","alpha_from_to":0.667}}
  - {{"type":"offset","anchor":"绿木块","offset_m":[0.05,0,0]}}
  - {{"type":"nearest_free","anchor":"胡萝卜","surface":"small_desk|worktable"}}
  - {{"type":"extreme_empty","surface":"small_desk|worktable","relation":"nearest|farthest|leftmost|rightmost|frontmost|backmost|left_front|right_front|left_back|right_back","reference_frame":"table_xy"}}

  Supported orientation expressions:
  - {{"type":"keep_current"}}
  - {{"type":"long_axis_toward","target":"笔筒","semantic_axis":"tip|long_axis"}}
  - {{"type":"long_axis_parallel_to","object_a":"红薯片","object_b":"笔筒"}}
  - {{"type":"upright|lay_flat|rotate_yaw|parallel_table_edge","yaw_deg":45}}

inside:
  {{"type":"inside","target":"笔筒","release":"drop"}}

on_top:
  {{"type":"on_top","target":"绿木块","stability_required":true,
    "fallback":{{"type":"beside","target":"绿木块","side":"right","clearance_m":0.015}}}}

beside:
  {{"type":"beside","target":"胶棒","side":"left|right|front|back|left_front|right_front|left_back|right_back","clearance_m":0.02,
    "long_axis":"preserve|parallel_table_edge|toward_target"}}

lean_against:
  {{"type":"lean_against","target":"笔筒","side":"right|left|front|back|left_front|right_front|left_back|right_back",
    "bottom_on":"table","lean_angle_deg":30,"long_axis":"toward_target"}}

between:
  {{"type":"between","object_a":"绿木块","object_b":"笔筒","face":"笔筒"}}

slot:
  {{"type":"slot","surface":"small_desk","slot":"slot_1|slot_2|...|slot_6",
    "orientation":{{"type":"upright|lay_flat|keep_current"}}}}
  For "竖着/立着/垂直放在 N 号位", include orientation={{"type":"upright"}} in the slot goal.
  For "横着/平放/躺着放在 N 号位", include orientation={{"type":"lay_flat"}} in the slot goal.

nearest_empty:
  {{"type":"nearest_empty","around":"小桌子","surface":"small_desk|worktable"}}

extreme_empty:
  {{"type":"extreme_empty","surface":"small_desk|worktable",
    "relation":"nearest|farthest|leftmost|rightmost|frontmost|backmost|left_front|right_front|left_back|right_back",
    "reference":"robot_base|surface_center|object name",
    "reference_frame":"table_xy"}}
  Use this for commands such as 桌子上最近的位置, 桌子上最远的位置, 最左/最右/最前/最后/左前/右前的空位.
  The default frame is table_xy: +X is front/前, +Y is left/左.

upright_in_place:
  {{"type":"upright_in_place"}}

rotate_in_place:
  {{"type":"rotate_in_place","yaw_deg":45}}

Important rules:
- Use one step per physical pick-place unless a later rotate/upright can be merged into the same target intent.
- Never put arithmetic expressions inside JSON. JSON numbers must be final numeric literals, not `0.14494 - 0.10`.
- For object-relative commands with an explicit direction and distance, prefer pose.position offset with a fully numeric offset_m. Examples: "网球后面10厘米" -> {{"type":"offset","anchor":"网球","offset_m":[-0.10,0,0]}}; "绿木块右边5厘米" -> {{"type":"offset","anchor":"绿木块","offset_m":[0,-0.05,0]}}; "胶棒左侧6厘米" -> {{"type":"offset","anchor":"胶棒","offset_m":[0,0.06,0]}}. Use beside only when the user says beside/next to/旁边 without a precise offset.
- For cleanup/整理/清理/all objects/所有物体/所有还没放好的东西, output one action="collection" step. Do not expand the collection into many pick_place pose steps yourself.
- For "桌子 1 到 6 号位" or "按从左到右放到 1 到 6 号位", use action="collection" with goal.type="slots", surface="small_desk", order="left_to_right".
- For "右半边/左半边/不要互相重叠" cleanup, use action="collection" with goal.type="right_half" or equivalent cleanup goal, not hand-written absolute coordinates.
- For "挡路的东西" without naming a concrete blocker, do not emit a speculative blocker move step. Emit the main goal; the local planner will handle feasibility.
- For line-ratio/exact-coordinate commands, compute coordinates from scene.object.position_xyz_m and scene.small_desk_slots, then output pose.position absolute_xyz or linear_interpolation.
- For "A 和 B 连线上靠近 B 三分之一", use pose.position linear_interpolation with alpha_from_to=0.667 from A to B.
- For "A 和 B 中间", use goal.type="between" with object_a=A and object_b=B. Add face=B if the object tip/long axis should point to B.
- For vague "靠着/靠在", prefer goal.type="lean_against" rather than "beside".
- For "叠到/放到上面，如果不稳就放旁边", use on_top with stability_required=true and fallback=beside.
- For "小桌子上最近的空位", use nearest_empty with surface="small_desk".
- For "桌子上最近/最远/最左/最右/左前/右前的位置", use extreme_empty. Do not encode 最远 as nearest_empty.
- For "竖着放在 N 号位", use slot with orientation={{"type":"upright"}}, not a plain slot.
- For "横着/平放/躺着放在 N 号位", use slot with orientation={{"type":"lay_flat"}}, not a plain slot.
- If an instruction is physically suspicious, still output the intended constraints and add an assumption/warning; do not fabricate a success.

Example 1:
{{
  "schema_version": "{LLM_PLAN_SCHEMA_VERSION}",
  "user_command": "把网球扔进笔筒，然后把笔靠在笔筒右侧",
  "steps": [
    {{"action":"pick_place","object":"网球","goal":{{"type":"inside","target":"笔筒","release":"drop"}}}},
    {{"action":"pick_place","object":"笔","goal":{{"type":"lean_against","target":"笔筒","side":"right","bottom_on":"table","lean_angle_deg":30,"long_axis":"toward_target"}}}}
  ]
}}

Example 2:
{{
  "schema_version": "{LLM_PLAN_SCHEMA_VERSION}",
  "user_command": "把胶棒叠到绿木块上面，如果支撑不稳就放到绿木块旁边",
  "steps": [
    {{"action":"pick_place","object":"胶棒","goal":{{"type":"on_top","target":"绿木块","stability_required":true,"fallback":{{"type":"beside","target":"绿木块","side":"right","clearance_m":0.015,"long_axis":"parallel_table_edge"}}}}}}
  ]
}}

Example 3:
{{
  "schema_version": "{LLM_PLAN_SCHEMA_VERSION}",
  "user_command": "笔竖着放在1号位",
  "steps": [
    {{"action":"pick_place","object":"笔","goal":{{"type":"slot","surface":"small_desk","slot":"slot_1","orientation":{{"type":"upright"}}}}}}
  ]
}}
"""


def _external_goal_to_internal_step(step: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(step, dict):
        raise ValueError("Each LLM step must be an object")
    action = str(step.get("action", "pick_place"))
    if action == "exchange":
        return {
            "kind": "multi_step",
            "operator": "exchange",
            "source_ref": step["object_a"],
            "other_ref": step["object_b"],
            "avoid_refs": list(step.get("avoid", []) or []),
            "notes": "external LLM exchange",
        }
    if action == "collection":
        goal = step.get("goal", {})
        if not isinstance(goal, dict):
            raise ValueError("collection.goal must be an object")
        if goal.get("type") == "slots":
            return {
                "kind": "multi_step",
                "operator": "collection_slots",
                "source_refs": list(step.get("objects", []) if step.get("objects") != "all_movable" else []),
                "slot_names": [f"slot_{idx}" for idx in range(1, 7)],
                "notes": "external LLM collection slots",
            }
        return {
            "kind": "multi_step",
            "operator": "collection_right_half",
            "source_refs": list(step.get("objects", []) if step.get("objects") != "all_movable" else []),
            "target_surface": str(goal.get("surface", "worktable")),
            "avoid_overlap": bool(goal.get("avoid_overlap", True)),
            "notes": "external LLM collection cleanup",
        }
    if action != "pick_place":
        raise ValueError(f"Unsupported LLM action: {action!r}")

    source_ref = step.get("object", step.get("source_ref"))
    if source_ref is None:
        raise ValueError("pick_place step is missing object")
    goal = step.get("goal")
    if not isinstance(goal, dict):
        raise ValueError("pick_place step is missing goal object")
    goal_type = str(goal.get("type", "")).strip()
    if goal_type in {"pose", "pose_goal", "target_pose"} or "position" in goal or "target_pose" in goal:
        pose_goal = dict(goal.get("target_pose") or goal)
        pose_goal.pop("type", None)
        return {
            "kind": "single_step",
            "operator": "pose_goal",
            "source_ref": source_ref,
            "pose_goal": pose_goal,
            "notes": "external LLM generalized pose goal",
        }
    if goal_type == "inside":
        return {"kind": "single_step", "operator": "inside", "source_ref": source_ref, "target_ref": goal["target"]}
    if goal_type in {"on_top", "on"}:
        fallback = goal.get("fallback")
        if bool(goal.get("stability_required", False)) and isinstance(fallback, dict):
            return {
                "kind": "single_step",
                "operator": "on_or_side_fallback",
                "source_ref": source_ref,
                "target_ref": goal["target"],
                "fallback_direction": str(fallback.get("side", fallback.get("direction", "right"))),
                "fallback_distance_m": float(fallback.get("clearance_m", 0.015)),
                "fallback_long_axis": fallback.get("long_axis"),
            }
        return {
            "kind": "single_step",
            "operator": "on",
            "source_ref": source_ref,
            "target_ref": goal["target"],
            "orientation_policy": goal.get("orientation_policy"),
            "anchor": "top_center",
        }
    if goal_type in {"beside", "relative"}:
        return {
            "kind": "single_step",
            "operator": "relative",
            "source_ref": source_ref,
            "target_ref": goal["target"],
            "direction": str(goal.get("side", goal.get("direction", "right"))),
            "distance_m": float(goal.get("clearance_m", goal.get("distance_m", 0.02))),
            "strict_distance": bool(goal.get("strict_distance", False)),
            "parallel_to": "desk_edge" if goal.get("long_axis") == "parallel_table_edge" else None,
            "long_axis": goal.get("long_axis"),
        }
    if goal_type == "lean_against":
        return {
            "kind": "single_step",
            "operator": "lean_against",
            "source_ref": source_ref,
            "target_ref": goal["target"],
            "side": str(goal.get("side", "right")),
            "lean_angle_deg": float(goal.get("lean_angle_deg", 30.0)),
            "bottom_on": str(goal.get("bottom_on", "table")),
        }
    if goal_type == "between":
        return {
            "kind": "single_step",
            "operator": "between",
            "source_ref": source_ref,
            "object_a_ref": goal["object_a"],
            "object_b_ref": goal["object_b"],
            "face_ref": goal.get("face"),
        }
    if goal_type == "slot":
        raw_slot = str(goal.get("slot", "slot_1"))
        orientation = goal.get("orientation")
        if isinstance(orientation, dict) or isinstance(orientation, str):
            return {
                "kind": "single_step",
                "operator": "pose_goal",
                "source_ref": source_ref,
                "pose_goal": {
                    "position": {"type": "slot", "surface": str(goal.get("surface", "small_desk")), "slot": raw_slot},
                    "orientation": orientation,
                },
                "notes": "external LLM slot with explicit orientation",
            }
        return {"kind": "single_step", "operator": "desk_slot", "source_ref": source_ref, "slot_name": raw_slot}
    if goal_type == "nearest_empty":
        return {
            "kind": "single_step",
            "operator": "nearest_empty",
            "source_ref": source_ref,
            "target_ref": goal.get("around", goal.get("target", "desk")),
            "surface": str(goal.get("surface", "worktable")),
        }
    if goal_type in {"extreme_empty", "empty_extreme", "farthest_empty", "leftmost_empty", "rightmost_empty"}:
        relation = str(goal.get("relation", "") or "").strip()
        if not relation:
            relation = {
                "farthest_empty": "farthest",
                "leftmost_empty": "leftmost",
                "rightmost_empty": "rightmost",
            }.get(goal_type, "nearest")
        return {
            "kind": "single_step",
            "operator": "extreme_empty",
            "source_ref": source_ref,
            "target_ref": goal.get("around", goal.get("target", goal.get("reference", "desk"))),
            "surface": str(goal.get("surface", "worktable")),
            "relation": relation,
            "reference": goal.get("reference"),
            "reference_frame": str(goal.get("reference_frame", "table_xy")),
        }
    if goal_type == "upright_in_place":
        return {"kind": "single_step", "operator": "upright_in_place", "source_ref": source_ref}
    if goal_type == "rotate_in_place":
        return {
            "kind": "single_step",
            "operator": "rotate_in_place",
            "source_ref": source_ref,
            "yaw_deg": float(goal.get("yaw_deg", 0.0)),
        }
    raise ValueError(f"Unsupported LLM goal.type: {goal_type!r}")


def _resolved_object_id_or_none(scene: SceneState, value: Any, *, role: str) -> str | None:
    if value is None:
        return None
    try:
        return str(resolve_object_ref(scene, str(value), role=role)["object_id"])
    except Exception:
        return None


def _collection_default_objects(scene: SceneState, *, slots: bool, order: str | None) -> list[SceneObject]:
    objects = list(_movable_scene_objects(scene))
    if slots and len(objects) > 6:
        # The physical small desk has six numbered slots. In this tabletop setup the
        # pen is normally paired with the pen holder, so do not let an unqualified
        # "all objects to slots 1-6" consume a desk slot with the pen.
        without_pen = [obj for obj in objects if obj.spec_name != "bi"]
        if len(without_pen) >= 6:
            objects = without_pen
    if str(order or "").strip() == "left_to_right":
        objects = sorted(objects, key=lambda obj: -float(obj.position[1]))
    return objects[:6] if slots else objects


def _reorder_dependent_support_moves(scene: SceneState, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reordered = list(steps)
    i = 0
    while i < len(reordered):
        step = reordered[i]
        op = str(step.get("operator", ""))
        if op not in {"on", "on_or_side_fallback"}:
            i += 1
            continue
        support_id = _resolved_object_id_or_none(scene, step.get("target_ref"), role="support_dependency")
        source_id = _resolved_object_id_or_none(scene, step.get("source_ref"), role="source_dependency")
        if support_id is None or source_id is None or support_id == source_id:
            i += 1
            continue
        move_idx = None
        for j in range(i + 1, len(reordered)):
            later_source = _resolved_object_id_or_none(scene, reordered[j].get("source_ref"), role="later_source_dependency")
            if later_source == support_id:
                move_idx = j
                break
        if move_idx is None:
            i += 1
            continue
        support_move = reordered.pop(move_idx)
        reordered.insert(i, support_move)
        i += 2
    return reordered


def external_llm_plan_to_internal(plan: dict[str, Any], scene: SceneState) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("LLM plan must be a JSON object")
    version = str(plan.get("schema_version", ""))
    if version and version != LLM_PLAN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version {version!r}; expected {LLM_PLAN_SCHEMA_VERSION!r}")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("LLM plan must contain a non-empty steps list")
    internal_steps = []
    for step in steps:
        internal_step = _external_goal_to_internal_step(step)
        if internal_step.get("operator") in {"collection_slots", "collection_right_half"} and not internal_step.get("source_refs"):
            goal = step.get("goal", {}) if isinstance(step.get("goal"), dict) else {}
            objects = _collection_default_objects(
                scene,
                slots=internal_step.get("operator") == "collection_slots",
                order=str(goal.get("order", "")),
            )
            internal_step["source_refs"] = [obj.object_id for obj in objects]
        internal_steps.append(internal_step)
    internal_steps = _reorder_dependent_support_moves(scene, internal_steps)
    if len(internal_steps) == 1:
        return internal_steps[0]
    return {
        "kind": "multi_step_sequence",
        "operator": "sequence",
        "steps": internal_steps,
        "notes": "external LLM structured plan",
    }


def _bool_from_any(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def _llm_provider_defaults(provider: str) -> tuple[str, str, str]:
    provider = str(provider or "mock").strip().lower()
    if provider == "deepseek":
        return "deepseek-v4-flash", "https://api.deepseek.com", "DEEPSEEK_API_KEY"
    return "gpt-4.1-mini", "https://api.openai.com/v1", "OPENAI_API_KEY"


def _llm_chat_endpoint(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    if not base:
        raise ValueError("LLM api_base is empty")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _json_object_from_text(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"LLM response does not contain a JSON object: {raw[:300]!r}")
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                data = json.loads(raw[start : idx + 1])
                if not isinstance(data, dict):
                    raise ValueError("LLM JSON root must be an object")
                return data
    raise ValueError(f"Could not parse balanced JSON object from LLM response: {raw[:500]!r}")


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _llm_prompt_messages(command: str, scene: SceneState) -> list[dict[str, str]]:
    system = (
        "You are the planner front-end for an RM75 tabletop pick-place robot. "
        "Convert the user's natural-language command into exactly one valid JSON object that follows the interface below. "
        "Do not output markdown. Prefer goal.type='pose' for geometric placement/orientation constraints. "
        "Resolve pronouns and omitted subjects from the command context. If the user says 'then rotate 45 degrees' after a placement, "
        "merge that rotation into the same object's target pose unless the command explicitly asks for a second pick.\n\n"
        + _llm_pick_place_interface_text()
    )
    user_payload = {
        "user_command": command,
        "scene_context": scene.context(),
        "output_requirement": "Return only the JSON plan object. No explanation.",
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def call_external_llm_plan(args, command: str, scene: SceneState, out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    provider = str(getattr(args, "llm_provider", "mock") or "mock").strip().lower()
    if provider in {"", "mock", "local_mock"}:
        raise ValueError("call_external_llm_plan called with mock provider")
    default_model, default_base, default_key_env = _llm_provider_defaults(provider)
    model = str(getattr(args, "llm_model", None) or DEFAULT_LLM_MODEL or default_model)
    api_base = str(getattr(args, "llm_api_base", None) or DEFAULT_LLM_API_BASE or default_base)
    key_env = str(getattr(args, "llm_api_key_env", None) or DEFAULT_LLM_API_KEY_ENV or default_key_env)
    api_key = os.environ.get(key_env)
    if not api_key:
        raise RuntimeError(f"LLM API key env var {key_env!r} is not set")
    messages = _llm_prompt_messages(command, scene)
    payload_base: dict[str, Any] = {
        "model": model,
        "stream": False,
        "temperature": float(getattr(args, "llm_temperature", 0.0) or 0.0),
        "max_tokens": int(getattr(args, "llm_max_tokens", 4096) or 4096),
    }
    if _bool_from_any(getattr(args, "llm_json_mode", True), True):
        payload_base["response_format"] = {"type": "json_object"}
    if provider == "deepseek":
        thinking_enabled = _bool_from_any(getattr(args, "llm_thinking", False), False)
        payload_base["thinking"] = {"type": "enabled" if thinking_enabled else "disabled"}
        if thinking_enabled:
            payload_base["reasoning_effort"] = str(getattr(args, "llm_reasoning_effort", "high") or "high")

    out_dir.mkdir(parents=True, exist_ok=True)
    repair_attempts = max(0, int(getattr(args, "llm_repair_attempts", 1) or 0))
    response: dict[str, Any] | None = None
    elapsed_ms_total = 0.0
    content = ""
    finish_reason = None
    raw_plan: dict[str, Any] | None = None
    parse_error: Exception | None = None

    def _write_attempt_file(base_name: str, attempt_idx: int, text: str) -> Path:
        base_path = Path(base_name)
        suffix = "" if attempt_idx == 0 else f"_repair_{attempt_idx}"
        path = out_dir / f"{base_path.stem}{suffix}{base_path.suffix}"
        path.write_text(text, encoding="utf-8")
        return path

    for attempt_idx in range(repair_attempts + 1):
        payload = {**payload_base, "messages": messages}
        safe_payload = json.loads(json.dumps(payload, ensure_ascii=False))
        _write_attempt_file("llm_request.json", attempt_idx, json.dumps(safe_payload, ensure_ascii=False, indent=2))
        req = urllib.request.Request(
            _llm_chat_endpoint(api_base),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=float(getattr(args, "llm_timeout_s", 60.0) or 60.0)) as resp:
                raw_bytes = resp.read()
                response = json.loads(raw_bytes.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            _write_attempt_file("llm_error.txt", attempt_idx, body)
            raise RuntimeError(f"LLM HTTP {exc.code}: {body[:1200]}")
        elapsed_ms_total += (time.perf_counter() - t0) * 1000.0
        _write_attempt_file("llm_raw_response.json", attempt_idx, json.dumps(response, ensure_ascii=False, indent=2))
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response has no choices: {response}")
        message = choices[0].get("message") or {}
        finish_reason = choices[0].get("finish_reason")
        content = _message_content_to_text(message.get("content"))
        _write_attempt_file("llm_response_text.txt", attempt_idx, content)
        try:
            raw_plan = _json_object_from_text(content)
            break
        except Exception as exc:
            parse_error = exc
            if attempt_idx >= repair_attempts:
                raise
            messages = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "The previous response was not valid JSON. "
                        f"Parser error: {type(exc).__name__}: {exc}. "
                        "Return one corrected JSON object only. Use numeric literals only; "
                        "do not put arithmetic expressions such as `0.14 - 0.10` inside JSON."
                    ),
                },
            ]
    if raw_plan is None:
        raise RuntimeError(f"LLM response could not be parsed: {parse_error!r}")
    (out_dir / "llm_plan_raw.json").write_text(json.dumps(raw_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {
        "provider": provider,
        "model": model,
        "api_base": api_base,
        "api_key_env": key_env,
        "elapsed_ms": float(elapsed_ms_total),
        "request_file": str(out_dir / "llm_request.json"),
        "raw_response_file": str(out_dir / "llm_raw_response.json"),
        "response_text_file": str(out_dir / "llm_response_text.txt"),
        "raw_plan_file": str(out_dir / "llm_plan_raw.json"),
        "usage": response.get("usage") if response is not None else None,
        "finish_reason": finish_reason,
        "repair_attempts_used": int(max(0, len([p for p in out_dir.glob("llm_response_text_repair_*.txt")]))),
    }
    return raw_plan, info


@dataclass
class ResolvedStep:
    index: int
    source_id: str
    operator: str
    target_object_id: str
    target_pose: np.ndarray
    place_mode: str
    primitive: str
    description: str
    mock_llm_step: dict[str, Any]
    warnings: list[str]


def _desk_object(scene: SceneState) -> SceneObject:
    desks = scene.find_by_spec("desk")
    if not desks:
        raise ValueError("Scene has no desk object; surface goals need a support object")
    return desks[0]


def _support_top_z(scene: SceneState, support_id: str) -> float:
    support = scene.get(support_id)
    return _world_z_bounds(support.T_world_obj, support.spec_name)[1]


def _preserve_pose_at_position(source: SceneObject, position: np.ndarray, scene: SceneState, support_id: str) -> np.ndarray:
    T = np.asarray(source.T_world_obj, dtype=np.float32).reshape(4, 4).copy()
    T[:3, 3] = np.asarray(position, dtype=np.float32).reshape(3)
    support = scene.get(support_id)
    support_z = _scene_object_surface_z(scene) if support.spec_name == "desk" else _object_support_top_z(scene, support)
    return _adjust_bottom_to_z(T, source.spec_name, support_z, margin=0.002)


def _upright_rotation(source: SceneObject) -> np.ndarray:
    mins, maxs = _mesh_bounds_scaled(source.spec_name)
    extents = np.asarray(maxs - mins, dtype=np.float32)
    long_idx = int(np.argmax(extents))
    current_R = np.asarray(source.T_world_obj[:3, :3], dtype=np.float32).reshape(3, 3)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cols = [None, None, None]
    cols[long_idx] = z_axis

    best_idx = 0 if long_idx != 0 else 1
    best_proj = None
    best_norm = -1.0
    for idx in range(3):
        if idx == long_idx:
            continue
        proj = current_R[:, idx].copy()
        proj[2] = 0.0
        norm = float(np.linalg.norm(proj))
        if norm > best_norm:
            best_norm = norm
            best_proj = proj
            best_idx = idx
    if best_proj is None or best_norm <= 1e-6:
        best_proj = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        best_proj = (best_proj / best_norm).astype(np.float32)
    cols[best_idx] = best_proj
    remaining_idx = [idx for idx in range(3) if cols[idx] is None][0]
    if (long_idx, best_idx, remaining_idx) in {(0, 1, 2), (1, 2, 0), (2, 0, 1)}:
        cols[remaining_idx] = np.cross(cols[long_idx], cols[best_idx]).astype(np.float32)
    else:
        cols[remaining_idx] = np.cross(cols[best_idx], cols[long_idx]).astype(np.float32)
    R = np.column_stack(cols).astype(np.float32)
    if np.linalg.det(R) < 0:
        R[:, remaining_idx] *= -1.0
    return R


def _horizontal_rotation(source: SceneObject) -> np.ndarray:
    mins, maxs = _mesh_bounds_scaled(source.spec_name)
    extents = np.asarray(maxs - mins, dtype=np.float32)
    long_idx = int(np.argmax(extents))
    current_R = np.asarray(source.T_world_obj[:3, :3], dtype=np.float32).reshape(3, 3)
    long_axis = current_R[:, long_idx].copy()
    long_axis[2] = 0.0
    long_norm = float(np.linalg.norm(long_axis))
    if long_norm <= 1e-6:
        long_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        long_axis = (long_axis / long_norm).astype(np.float32)

    non_long = [idx for idx in range(3) if idx != long_idx]
    up_idx = min(non_long, key=lambda idx: float(extents[idx]))
    remaining_idx = [idx for idx in range(3) if idx not in {long_idx, up_idx}][0]
    cols = [None, None, None]
    cols[long_idx] = long_axis
    cols[up_idx] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if (long_idx, up_idx, remaining_idx) in {(0, 1, 2), (1, 2, 0), (2, 0, 1)}:
        cols[remaining_idx] = np.cross(cols[long_idx], cols[up_idx]).astype(np.float32)
    else:
        cols[remaining_idx] = np.cross(cols[up_idx], cols[long_idx]).astype(np.float32)
    R = np.column_stack(cols).astype(np.float32)
    if np.linalg.det(R) < 0:
        R[:, remaining_idx] *= -1.0
    return R


def _apply_orientation_policy(source: SceneObject, policy: str | None) -> np.ndarray:
    T = np.asarray(source.T_world_obj, dtype=np.float32).reshape(4, 4).copy()
    if policy == "horizontal":
        T[:3, :3] = _horizontal_rotation(source)
    elif policy == "vertical":
        T[:3, :3] = _upright_rotation(source)
    return T


def _set_long_axis_xy(T_world_obj: np.ndarray, spec_name: str, direction_xy: np.ndarray) -> np.ndarray:
    T = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4).copy()
    mins, maxs = _mesh_bounds_scaled(spec_name)
    extents = np.asarray(maxs - mins, dtype=np.float32)
    long_idx = int(np.argmax(extents))
    direction = np.asarray(direction_xy, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return T
    long_axis = np.array([direction[0] / norm, direction[1] / norm, 0.0], dtype=np.float32)
    non_long = [idx for idx in range(3) if idx != long_idx]
    up_idx = min(non_long, key=lambda idx: float(extents[idx]))
    remaining_idx = [idx for idx in range(3) if idx not in {long_idx, up_idx}][0]
    cols = [None, None, None]
    cols[long_idx] = long_axis
    cols[up_idx] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if (long_idx, up_idx, remaining_idx) in {(0, 1, 2), (1, 2, 0), (2, 0, 1)}:
        cols[remaining_idx] = np.cross(cols[long_idx], cols[up_idx]).astype(np.float32)
    else:
        cols[remaining_idx] = np.cross(cols[up_idx], cols[long_idx]).astype(np.float32)
    R = np.column_stack(cols).astype(np.float32)
    if np.linalg.det(R) < 0:
        R[:, remaining_idx] *= -1.0
    T[:3, :3] = R
    return T


def _set_long_axis_world(T_world_obj: np.ndarray, spec_name: str, direction_world: np.ndarray, side_axis_xy: np.ndarray | None = None) -> np.ndarray:
    T = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4).copy()
    mins, maxs = _mesh_bounds_scaled(spec_name)
    extents = np.asarray(maxs - mins, dtype=np.float32)
    long_idx = int(np.argmax(extents))
    direction = np.asarray(direction_world, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return T
    long_axis = (direction / norm).astype(np.float32)

    non_long = [idx for idx in range(3) if idx != long_idx]
    if side_axis_xy is None:
        side_axis = np.array([-long_axis[1], long_axis[0], 0.0], dtype=np.float32)
    else:
        side_xy = np.asarray(side_axis_xy, dtype=np.float32).reshape(2)
        side_axis = np.array([side_xy[0], side_xy[1], 0.0], dtype=np.float32)
    side_axis -= long_axis * float(np.dot(side_axis, long_axis))
    side_norm = float(np.linalg.norm(side_axis))
    if side_norm <= 1e-6:
        side_axis = np.cross(long_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
        side_norm = float(np.linalg.norm(side_axis))
    if side_norm <= 1e-6:
        side_axis = np.cross(long_axis, np.array([1.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
        side_norm = float(np.linalg.norm(side_axis))
    if side_norm <= 1e-6:
        return T
    side_axis = (side_axis / side_norm).astype(np.float32)

    side_idx = non_long[0]
    remaining_idx = non_long[1]
    cols = [None, None, None]
    cols[long_idx] = long_axis
    cols[side_idx] = side_axis
    if (long_idx, side_idx, remaining_idx) in {(0, 1, 2), (1, 2, 0), (2, 0, 1)}:
        cols[remaining_idx] = np.cross(cols[long_idx], cols[side_idx]).astype(np.float32)
    else:
        cols[remaining_idx] = np.cross(cols[side_idx], cols[long_idx]).astype(np.float32)
    R = np.column_stack(cols).astype(np.float32)
    if np.linalg.det(R) < 0:
        R[:, remaining_idx] *= -1.0
    T[:3, :3] = R
    return T


def _xy_half_extent_along(T_world_obj: np.ndarray, spec_name: str, direction_xy: np.ndarray) -> float:
    direction = np.asarray(direction_xy, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return _footprint_radius(spec_name)
    direction = direction / norm
    vertices = _mesh_vertices_scaled(spec_name)
    T = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
    xy = (T[:3, :3] @ vertices.T).T[:, :2] + T[:2, 3]
    projection = xy @ direction
    return 0.5 * float(np.max(projection) - np.min(projection))


def _orient_long_axis_toward(T_world_obj: np.ndarray, source: SceneObject, target: SceneObject) -> np.ndarray:
    T = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
    direction = np.asarray(target.position[:2] - T[:2, 3], dtype=np.float32)
    return _set_long_axis_xy(T, source.spec_name, direction)


def _orient_parallel_to_desk_edge(scene: SceneState, T_world_obj: np.ndarray, source: SceneObject) -> np.ndarray:
    desk = _desk_object(scene)
    candidates = []
    for idx in (0, 2):
        vec = np.asarray(desk.T_world_obj[:3, idx], dtype=np.float32).reshape(3)
        vec[2] = 0.0
        norm = float(np.linalg.norm(vec[:2]))
        if norm > 1e-6:
            candidates.append(vec[:2] / norm)
    direction = candidates[0] if candidates else np.array([1.0, 0.0], dtype=np.float32)
    return _set_long_axis_xy(T_world_obj, source.spec_name, direction)


def _surface_empty_pose(
    scene: SceneState,
    source: SceneObject,
    *,
    anchor_xy: np.ndarray,
    prefer_near: bool = True,
    right_half: bool = False,
) -> tuple[np.ndarray, str]:
    desk = _desk_object(scene)
    surface_z = _scene_object_surface_z(scene)
    source_r = _footprint_radius(source.spec_name)
    object_xy = [
        np.asarray(obj.position[:2], dtype=np.float32).reshape(2)
        for obj in scene.objects.values()
        if obj.spec_name != "desk"
    ]
    all_xy = np.vstack(object_xy) if object_xy else np.asarray(anchor_xy, dtype=np.float32).reshape(1, 2)
    min_xy = np.min(all_xy, axis=0) - np.array([0.10, 0.10], dtype=np.float32)
    max_xy = np.max(all_xy, axis=0) + np.array([0.10, 0.10], dtype=np.float32)
    if right_half:
        center_x = float(np.mean(all_xy[:, 0]))
        min_xy[0] = max(float(min_xy[0]), center_x + 0.03)
        max_xy[0] = max(float(max_xy[0]), center_x + 0.20)
    anchor = np.asarray(anchor_xy, dtype=np.float32).reshape(2)
    best_xy = anchor.copy()
    best_score = -float("inf")
    for x in np.linspace(float(min_xy[0]), float(max_xy[0]), 13):
        for y in np.linspace(float(min_xy[1]), float(max_xy[1]), 11):
            xy = np.array([x, y], dtype=np.float32)
            min_gap = float("inf")
            for obj in scene.objects.values():
                if obj.object_id == source.object_id or obj.spec_name == "desk":
                    continue
                gap = float(np.linalg.norm(xy - obj.position[:2])) - source_r - _footprint_radius(obj.spec_name)
                min_gap = min(min_gap, gap)
            dist = float(np.linalg.norm(xy - anchor))
            score = min_gap - (0.65 * dist if prefer_near else 0.05 * dist)
            if right_half:
                score += 0.15 * float(xy[0])
            if score > best_score:
                best_score = score
                best_xy = xy
    return _pose_xy_on_surface(source.T_world_obj, source.spec_name, best_xy, surface_z), desk.object_id


def _reference_frame_axes(scene: SceneState, *, surface: str = "worktable") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return origin, forward, right axes in XY. Convention: +X front, +Y left."""
    del scene, surface
    origin = DEFAULT_ROBOT_BASE_WORLD_XYZ[:2].astype(np.float32)
    forward = np.array([1.0, 0.0], dtype=np.float32)
    right = np.array([0.0, -1.0], dtype=np.float32)
    return origin, forward, right


def _worktable_empty_candidate_xy(scene: SceneState, *, right_half: bool = False) -> list[np.ndarray]:
    object_xy = [
        np.asarray(obj.position[:2], dtype=np.float32).reshape(2)
        for obj in scene.objects.values()
        if obj.spec_name != "desk"
    ]
    if object_xy:
        all_xy = np.vstack(object_xy)
        min_xy = np.min(all_xy, axis=0) - np.array([0.10, 0.10], dtype=np.float32)
        max_xy = np.max(all_xy, axis=0) + np.array([0.10, 0.10], dtype=np.float32)
    else:
        min_xy = np.array([-0.45, -0.25], dtype=np.float32)
        max_xy = np.array([0.15, 0.25], dtype=np.float32)
    if right_half:
        center_x = float(np.mean(all_xy[:, 0])) if object_xy else float(np.mean([min_xy[0], max_xy[0]]))
        min_xy[0] = max(float(min_xy[0]), center_x + 0.03)
        max_xy[0] = max(float(max_xy[0]), center_x + 0.20)
    candidates: list[np.ndarray] = []
    for x in np.linspace(float(min_xy[0]), float(max_xy[0]), 13):
        for y in np.linspace(float(min_xy[1]), float(max_xy[1]), 11):
            candidates.append(np.array([x, y], dtype=np.float32))
    return candidates


def _empty_pose_by_extreme(
    scene: SceneState,
    source: SceneObject,
    *,
    surface: str = "worktable",
    relation: str = "nearest",
    reference_xy: np.ndarray | None = None,
    reference_frame: str = "table_xy",
) -> tuple[np.ndarray, str, list[str]]:
    surface = "small_desk" if surface == "small_desk" else "worktable"
    relation = str(relation or "nearest").strip().lower()
    aliases = {
        "near": "nearest",
        "closest": "nearest",
        "far": "farthest",
        "furthest": "farthest",
        "left": "leftmost",
        "right": "rightmost",
        "front": "frontmost",
        "back": "backmost",
        "behind": "backmost",
        "front_left": "left_front",
        "front_right": "right_front",
        "back_left": "left_back",
        "back_right": "right_back",
    }
    relation = aliases.get(relation, relation)
    if relation not in {"nearest", "farthest", "leftmost", "rightmost", "frontmost", "backmost", "left_front", "right_front", "left_back", "right_back"}:
        relation = "nearest"

    source_r = _footprint_radius(source.spec_name)
    if surface == "small_desk":
        support_z = _desk_slot_support_z(_desk_object(scene))
        support_id = _desk_object(scene).object_id
        candidates = _small_desk_candidate_xy(scene)

        def on_surface(obj: SceneObject) -> bool:
            return _object_is_on_small_desk(scene, obj)
    else:
        support_z = _scene_object_surface_z(scene)
        support_id = _desk_object(scene).object_id
        candidates = _worktable_empty_candidate_xy(scene)

        def on_surface(obj: SceneObject) -> bool:
            return obj.spec_name != "desk"

    origin, forward, right = _reference_frame_axes(scene, surface=surface)
    if reference_xy is not None:
        ref = np.asarray(reference_xy, dtype=np.float32).reshape(2)
    elif reference_frame in {"surface_center", "table_center", "desk_center"}:
        ref = _desk_object(scene).position[:2].astype(np.float32) if surface == "small_desk" else origin
    else:
        ref = origin

    best_xy: np.ndarray | None = None
    best_score = -float("inf")
    best_gap = -float("inf")
    for xy in candidates:
        min_gap = float("inf")
        for obstacle in scene.objects.values():
            if obstacle.object_id == source.object_id or obstacle.spec_name == "desk" or not on_surface(obstacle):
                continue
            gap = float(np.linalg.norm(xy - obstacle.position[:2])) - source_r - _footprint_radius(obstacle.spec_name)
            min_gap = min(min_gap, gap)
        if min_gap == float("inf"):
            min_gap = 0.25
        rel = xy - ref
        dist_robot = float(np.linalg.norm(xy - origin))
        if relation == "nearest":
            semantic = -dist_robot
        elif relation == "farthest":
            semantic = dist_robot
        elif relation == "rightmost":
            semantic = float(np.dot(rel, right))
        elif relation == "leftmost":
            semantic = -float(np.dot(rel, right))
        elif relation == "frontmost":
            semantic = float(np.dot(rel, forward))
        elif relation == "backmost":
            semantic = -float(np.dot(rel, forward))
        else:
            semantic = float(np.dot(rel, _table_direction_xy(relation)))
        score = 2.0 * semantic + 1.35 * min(float(min_gap), 0.08)
        if min_gap < 0.005:
            score -= 2.0 + 20.0 * (0.005 - min_gap)
        if min_gap > best_gap:
            best_gap = min_gap
        if score > best_score:
            best_score = score
            best_xy = xy
    if best_xy is None:
        best_xy = _desk_object(scene).position[:2].astype(np.float32) if surface == "small_desk" else origin
    warnings: list[str] = [f"empty extreme relation={relation} reference_frame=table_xy(+X front,+Y left)"]
    if best_gap < 0.012:
        warnings.append(f"selected empty pose has tight clearance: best_gap={best_gap:.3f}m")
    T_goal = _pose_xy_on_surface(source.T_world_obj, source.spec_name, best_xy, support_z)
    return T_goal, support_id, warnings


def _between_clearance_pose(
    scene: SceneState,
    source: SceneObject,
    a: SceneObject,
    b: SceneObject,
) -> tuple[np.ndarray, str, list[str]]:
    support_id = _desk_object(scene).object_id
    surface_z = _scene_object_surface_z(scene)
    source_r = _footprint_radius(source.spec_name)
    ideal = (a.position[:2] + b.position[:2]) * 0.5
    line = b.position[:2] - a.position[:2]
    line_norm = float(np.linalg.norm(line))
    if line_norm <= 1e-6:
        line_dir = np.array([1.0, 0.0], dtype=np.float32)
    else:
        line_dir = (line / line_norm).astype(np.float32)
    perp = np.array([-line_dir[1], line_dir[0]], dtype=np.float32)
    candidates = [ideal.astype(np.float32)]
    seen_candidates: set[tuple[float, float]] = set()

    def _add_candidate(xy: np.ndarray) -> None:
        xy = np.asarray(xy, dtype=np.float32).reshape(2)
        key = (round(float(xy[0]), 4), round(float(xy[1]), 4))
        if key in seen_candidates:
            return
        seen_candidates.add(key)
        candidates.append(xy)

    for offset in (0.025, 0.05, 0.075, 0.10, 0.14, 0.18, 0.22, 0.26):
        _add_candidate(ideal + perp * offset)
        _add_candidate(ideal - perp * offset)
    for along in np.linspace(-0.15, 0.15, 7):
        _add_candidate(ideal + line_dir * float(along))
        for offset in np.linspace(-0.25, 0.25, 11):
            _add_candidate(ideal + line_dir * float(along) + perp * float(offset))

    best_xy = candidates[0]
    best_gap = -float("inf")
    best_score = -float("inf")
    for xy in candidates:
        min_gap = float("inf")
        for obj in scene.objects.values():
            if obj.object_id == source.object_id or obj.spec_name == "desk":
                continue
            gap = float(np.linalg.norm(xy - obj.position[:2])) - source_r - _footprint_radius(obj.spec_name)
            min_gap = min(min_gap, gap)
        dist_ideal = float(np.linalg.norm(xy - ideal))
        dist_line = abs(float(np.cross(line_dir, xy - ideal)))
        score = 3.5 * min_gap - 0.35 * dist_ideal - 0.15 * dist_line
        if min_gap > 0.015:
            score += 2.0
        if source.spec_name == "hongshupian":
            if dist_ideal > 0.16:
                score -= 3.0 * (dist_ideal - 0.16)
            if float(xy[0]) < -0.33:
                score -= 4.0 + 8.0 * (-0.33 - float(xy[0]))
            # A vertical hongshupian hover pose near the robot-inner side is often
            # collision-free as a static joint state but unreachable for transport
            # planning. Use the same reachability guard as relative worktable goals.
            if float(xy[0]) < WORKTABLE_RELATIVE_INNER_GUARD_X_M:
                score -= 1.5 + 6.0 * (WORKTABLE_RELATIVE_INNER_GUARD_X_M - float(xy[0]))
                if -0.14 <= float(xy[1]) <= 0.10:
                    score -= 8.0 + 25.0 * (WORKTABLE_RELATIVE_INNER_GUARD_X_M - float(xy[0]))
        if score > best_score:
            best_score = score
            best_gap = min_gap
            best_xy = xy

    warnings: list[str] = []
    shift = float(np.linalg.norm(best_xy - ideal))
    if shift > 0.015:
        warnings.append(f"between target shifted {shift:.3f}m from midpoint for clearance")
    if best_gap < 0.0:
        warnings.append(f"between target still has rough overlap risk after scoring: min_gap={best_gap:.3f}m")
    elif best_gap < 0.015:
        warnings.append(f"between target clearance is tight after scoring: min_gap={best_gap:.3f}m")
    template = _default_surface_template(scene, source) if source.spec_name == "hongshupian" else source.T_world_obj
    return _pose_xy_on_surface(template, source.spec_name, best_xy, surface_z), support_id, warnings


def _blocking_object_for(scene: SceneState, source: SceneObject, target: SceneObject) -> SceneObject | None:
    a = source.position[:2]
    b = target.position[:2]
    ab = b - a
    denom = float(np.dot(ab, ab))
    best: tuple[float, SceneObject] | None = None
    for obj in _movable_scene_objects(scene):
        if obj.object_id in {source.object_id, target.object_id}:
            continue
        p = obj.position[:2]
        t = 0.0 if denom <= 1e-9 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        closest = a + t * ab
        dist = float(np.linalg.norm(p - closest))
        score = dist + 0.05 * abs(t - 0.5)
        if best is None or score < best[0]:
            best = (score, obj)
    return None if best is None else best[1]


def _inside_pose(scene: SceneState, source: SceneObject, target: SceneObject) -> tuple[np.ndarray, list[str]]:
    warnings = []
    target_axis = np.asarray(target.T_world_obj[:3, 1], dtype=np.float32)
    target_axis_norm = float(np.linalg.norm(target_axis))
    if target_axis_norm <= 1e-6:
        target_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        target_axis = target_axis / target_axis_norm
    if float(np.dot(target_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))) < 0.45:
        warnings.append(f"target {target.object_id} opening axis is not clearly upward")
    target_mins, target_maxs = _mesh_bounds_scaled(target.spec_name)
    source_mins, source_maxs = _mesh_bounds_scaled(source.spec_name)
    target_height = float(np.max(target_maxs - target_mins))
    source_radius = float(np.max(source_maxs - source_mins) * 0.5)
    local_offset = max(0.035, 0.50 * target_height + 0.20 * source_radius)
    T = np.asarray(source.T_world_obj, dtype=np.float32).reshape(4, 4).copy()
    T[:3, 3] = target.position + target_axis * float(local_offset)
    if source.spec_name == "bi":
        default_rule = get_place_rule("bi")
        if default_rule is not None and default_rule.object_pose_local is not None:
            local = np.eye(4, dtype=np.float32)
            local[:3, 3] = np.asarray(default_rule.object_pose_local.position, dtype=np.float32)
            local[:3, :3] = euler2mat(
                *np.deg2rad(np.asarray(default_rule.object_pose_local.rpy_deg, dtype=np.float32)),
                axes="sxyz",
            ).astype(np.float32)
            T = target.T_world_obj @ local
    return T.astype(np.float32), warnings


def _desk_slot_support_z(desk: SceneObject) -> float:
    try:
        _, top = _precise_world_z_bounds(desk.T_world_obj, desk.spec_name)
    except Exception:
        _, top = _world_z_bounds(desk.T_world_obj, desk.spec_name)
    return float(top)


def _slot_pose(scene: SceneState, source: SceneObject, slot_name: str) -> tuple[np.ndarray, str]:
    desk = _desk_object(scene)
    source_rule = get_place_rule(source.spec_name)
    if source_rule is not None and source_rule.slots:
        for slot in source_rule.slots:
            if slot.name != slot_name:
                continue
            local = np.eye(4, dtype=np.float32)
            local[:3, 3] = np.asarray(slot.object_pose_local.position, dtype=np.float32)
            local[:3, :3] = euler2mat(
                *np.deg2rad(np.asarray(slot.object_pose_local.rpy_deg, dtype=np.float32)),
                axes="sxyz",
            ).astype(np.float32)
            return (desk.T_world_obj @ local).astype(np.float32), desk.object_id
    slot_lookup = {name: (x, z) for name, (x, z) in DESK_SLOT_LAYOUT_XZ}
    if slot_name not in slot_lookup:
        raise ValueError(f"Unknown desk slot {slot_name!r}")
    x, z = slot_lookup[slot_name]
    local = np.eye(4, dtype=np.float32)
    local[:3, 3] = np.array([float(x), 0.08, float(z)], dtype=np.float32)
    T = np.asarray(source.T_world_obj, dtype=np.float32).reshape(4, 4).copy()
    T[:3, 3] = (desk.T_world_obj @ local)[:3, 3]
    return _pose_xy_on_surface(T, source.spec_name, T[:2, 3], _desk_slot_support_z(desk)), desk.object_id


def _default_surface_template(scene: SceneState, source: SceneObject) -> np.ndarray:
    T = np.asarray(source.T_world_obj, dtype=np.float32).reshape(4, 4).copy()
    source_rule = get_place_rule(source.spec_name)
    if source_rule is not None and source_rule.slots:
        slot = source_rule.slots[0]
        local_R = euler2mat(
            *np.deg2rad(np.asarray(slot.object_pose_local.rpy_deg, dtype=np.float32)),
            axes="sxyz",
        ).astype(np.float32)
        T[:3, :3] = (_desk_object(scene).T_world_obj[:3, :3] @ local_R).astype(np.float32)
    return T


def _desk_xy_gap(scene: SceneState, xy: np.ndarray, source_radius: float) -> float:
    desk = _desk_object(scene)
    desk_radius = _footprint_radius(desk.spec_name)
    return float(np.linalg.norm(np.asarray(xy, dtype=np.float32).reshape(2) - desk.position[:2])) - float(source_radius) - desk_radius


WORKTABLE_SMALL_DESK_MIN_GAP_M = 0.08
WORKTABLE_RELATIVE_INNER_GUARD_X_M = -0.24
MAX_POSE_GOAL_CLEARANCE_SHIFT_M = 0.05
MAX_RELATIVE_CLEARANCE_SHIFT_M = 0.08


def _pose_goal_clearance_adjusted_xy(
    scene: SceneState,
    source: SceneObject,
    nominal_xy: np.ndarray,
    *,
    surface: str,
) -> tuple[np.ndarray, list[str]]:
    xy0 = np.asarray(nominal_xy, dtype=np.float32).reshape(2)
    source_r = _footprint_radius(source.spec_name)
    surface = "small_desk" if surface == "small_desk" else "worktable"

    def _on_relevant_surface(obj: SceneObject) -> bool:
        if obj.spec_name == "desk":
            return False
        return _object_is_on_small_desk(scene, obj) if surface == "small_desk" else True

    def _gaps(xy: np.ndarray) -> tuple[float, float]:
        min_gap = float("inf")
        for obstacle in scene.objects.values():
            if obstacle.object_id == source.object_id or not _on_relevant_surface(obstacle):
                continue
            gap = float(np.linalg.norm(xy - obstacle.position[:2])) - source_r - _footprint_radius(obstacle.spec_name)
            min_gap = min(min_gap, gap)
        if min_gap == float("inf"):
            min_gap = 0.25
        desk_gap = 0.25 if surface == "small_desk" else _desk_xy_gap(scene, xy, source_r)
        return min_gap, desk_gap

    nominal_gap, nominal_desk_gap = _gaps(xy0)
    if nominal_gap >= 0.010 and nominal_desk_gap >= WORKTABLE_SMALL_DESK_MIN_GAP_M:
        return xy0, []

    candidates: list[np.ndarray] = [xy0]
    for radius in (0.015, 0.03, 0.05, 0.07, 0.09):
        for theta in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            candidates.append(xy0 + float(radius) * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32))

    best_xy = xy0
    best_score = -float("inf")
    best_gap = nominal_gap
    best_desk_gap = nominal_desk_gap
    for xy in candidates:
        min_gap, desk_gap = _gaps(xy)
        shift = float(np.linalg.norm(xy - xy0))
        score = 2.4 * min_gap - 1.8 * shift
        if min_gap > 0.012:
            score += 1.0
        if surface != "small_desk" and desk_gap < WORKTABLE_SMALL_DESK_MIN_GAP_M:
            score -= 4.0 + 20.0 * (WORKTABLE_SMALL_DESK_MIN_GAP_M - desk_gap)
        if shift > 0.075:
            score -= 2.5 * (shift - 0.075)
        if score > best_score:
            best_score = score
            best_xy = xy.astype(np.float32)
            best_gap = min_gap
            best_desk_gap = desk_gap

    warnings: list[str] = []
    shift = float(np.linalg.norm(best_xy - xy0))
    if shift > MAX_POSE_GOAL_CLEARANCE_SHIFT_M:
        warnings.append(
            f"rough clearance adjustment rejected: would shift pose target {shift:.3f}m from the requested semantic point"
        )
        best_xy = xy0
        best_gap = nominal_gap
        best_desk_gap = nominal_desk_gap
        shift = 0.0
    if shift > 0.010:
        warnings.append(f"pose target shifted {shift:.3f}m for rough clearance")
    if best_gap < 0.0:
        warnings.append(f"pose target still has rough overlap risk after shift: min_gap={best_gap:.3f}m")
    elif best_gap < 0.010:
        warnings.append(f"pose target clearance is tight after shift: min_gap={best_gap:.3f}m")
    if surface != "small_desk" and best_desk_gap < WORKTABLE_SMALL_DESK_MIN_GAP_M:
        warnings.append(f"pose target is close to small desk after shift: gap={best_desk_gap:.3f}m")
    return best_xy.astype(np.float32), warnings


def _worktable_surface_template(scene: SceneState, source: SceneObject) -> np.ndarray:
    if source.spec_name == "gluestick":
        return _apply_orientation_policy(source, "horizontal")
    return _default_surface_template(scene, source)


def _small_desk_candidate_xy(scene: SceneState) -> list[np.ndarray]:
    desk = _desk_object(scene)
    candidates: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()

    def _add(local_x: float, local_z: float) -> None:
        local = np.array([float(local_x), 0.08, float(local_z)], dtype=np.float32)
        world = (desk.T_world_obj[:3, :3] @ local) + desk.T_world_obj[:3, 3]
        xy = np.asarray(world[:2], dtype=np.float32).reshape(2)
        key = (round(float(xy[0]), 4), round(float(xy[1]), 4))
        if key in seen:
            return
        seen.add(key)
        candidates.append(xy)

    for _, (local_x, local_z) in DESK_SLOT_LAYOUT_XZ:
        _add(local_x, local_z)
    for local_x in np.linspace(-0.14, 0.14, 5):
        for local_z in np.linspace(-0.07, 0.07, 3):
            _add(float(local_x), float(local_z))
    return candidates


def _object_is_on_small_desk(scene: SceneState, obj: SceneObject) -> bool:
    if obj.spec_name == "desk":
        return True
    desk = _desk_object(scene)
    desk_radius = _footprint_radius(desk.spec_name)
    obj_radius = _footprint_radius(obj.spec_name)
    try:
        bottom, _ = _precise_world_z_bounds(obj.T_world_obj, obj.spec_name)
    except Exception:
        bottom, _ = _world_z_bounds(obj.T_world_obj, obj.spec_name)
    top_z = _desk_slot_support_z(desk)
    xy_gap = float(np.linalg.norm(obj.position[:2] - desk.position[:2])) - desk_radius - obj_radius
    return bool(xy_gap <= 0.08 and abs(float(bottom) - float(top_z)) <= 0.065)


def _small_desk_footprint_margin(scene: SceneState, T_world_obj: np.ndarray, spec_name: str) -> float:
    """Positive if the object's projected footprint fits inside the small desk."""
    desk = _desk_object(scene)
    desk_mins, desk_maxs = _mesh_bounds_scaled(desk.spec_name)
    pts_world = _world_points(T_world_obj, spec_name)
    R_world_desk = np.asarray(desk.T_world_obj[:3, :3], dtype=np.float32)
    p_world_desk = np.asarray(desk.T_world_obj[:3, 3], dtype=np.float32)
    pts_local = (R_world_desk.T @ (pts_world - p_world_desk).T).T
    margin_x = min(
        float(np.min(pts_local[:, 0]) - float(desk_mins[0])),
        float(float(desk_maxs[0]) - np.max(pts_local[:, 0])),
    )
    margin_z = min(
        float(np.min(pts_local[:, 2]) - float(desk_mins[2])),
        float(float(desk_maxs[2]) - np.max(pts_local[:, 2])),
    )
    return min(margin_x, margin_z)


def _small_desk_pose_from_xy(scene: SceneState, source: SceneObject, xy: np.ndarray) -> tuple[np.ndarray, str]:
    desk = _desk_object(scene)
    T_template = _default_surface_template(scene, source)
    return _pose_xy_on_surface(T_template, source.spec_name, xy, _desk_slot_support_z(desk)), desk.object_id


def _small_desk_empty_pose(
    scene: SceneState,
    source: SceneObject,
    *,
    anchor_xy: np.ndarray | None = None,
    direction_xy: np.ndarray | None = None,
    prefer_near: bool = True,
) -> tuple[np.ndarray, str]:
    source_r = _footprint_radius(source.spec_name)
    desk = _desk_object(scene)
    anchor = desk.position[:2] if anchor_xy is None else np.asarray(anchor_xy, dtype=np.float32).reshape(2)
    direction = None
    if direction_xy is not None:
        direction = np.asarray(direction_xy, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(direction))
        direction = None if norm <= 1e-6 else (direction / norm).astype(np.float32)

    best_xy: np.ndarray | None = None
    best_score = -float("inf")
    for xy in _small_desk_candidate_xy(scene):
        T_candidate, _ = _small_desk_pose_from_xy(scene, source, xy)
        desk_margin = _small_desk_footprint_margin(scene, T_candidate, source.spec_name)
        min_gap = float("inf")
        for obstacle in scene.objects.values():
            if obstacle.object_id == source.object_id or obstacle.spec_name == "desk":
                continue
            if not _object_is_on_small_desk(scene, obstacle):
                continue
            gap = float(np.linalg.norm(xy - obstacle.position[:2])) - source_r - _footprint_radius(obstacle.spec_name)
            min_gap = min(min_gap, gap)
        dist_anchor = float(np.linalg.norm(xy - anchor))
        score = 2.5 * min_gap - (0.55 if prefer_near else 0.10) * dist_anchor
        if direction is not None:
            projection = float(np.dot(xy - anchor, direction))
            score += 1.20 * projection
            if projection < 0.015:
                score -= 1.25
            rel = xy - anchor
            lateral = abs(float(direction[0] * rel[1] - direction[1] * rel[0]))
            score -= 0.55 * lateral
        if min_gap > 0.012:
            score += 1.0
        if desk_margin < 0.0:
            score -= 100.0 + 20.0 * abs(desk_margin)
        elif desk_margin < 0.008:
            score -= 2.0 * (0.008 - desk_margin)
        else:
            score += min(0.08, desk_margin)
        if score > best_score:
            best_score = score
            best_xy = xy
    if best_xy is None:
        best_xy = anchor
    return _small_desk_pose_from_xy(scene, source, best_xy)


def _worktable_relative_pose(
    scene: SceneState,
    source: SceneObject,
    target: SceneObject,
    direction: str,
    distance_m: float,
    *,
    strict_distance: bool = False,
) -> tuple[np.ndarray, str, list[str]]:
    desk = _desk_object(scene)
    vec = _table_direction_xy(direction)
    perp = np.array([-vec[1], vec[0]], dtype=np.float32)
    source_r = _footprint_radius(source.spec_name)
    target_r = _footprint_radius(target.spec_name)
    center_distance = float(distance_m) + source_r + target_r
    nominal = target.position[:2] + vec * center_distance

    candidates: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()

    def _add(xy: np.ndarray) -> None:
        xy = np.asarray(xy, dtype=np.float32).reshape(2)
        key = (round(float(xy[0]), 4), round(float(xy[1]), 4))
        if key in seen:
            return
        seen.add(key)
        candidates.append(xy)

    for scale in (0.75, 1.0, 1.25, 1.5):
        for offset in (-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10):
            _add(target.position[:2] + vec * (center_distance * scale) + perp * float(offset))
    for radius in (0.10, 0.14, 0.18, 0.23):
        for theta in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            _add(target.position[:2] + radius * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32))

    def _relative_clearance(xy: np.ndarray) -> tuple[float, float, bool]:
        xy = np.asarray(xy, dtype=np.float32).reshape(2)
        min_gap = float("inf")
        desk_gap = _desk_xy_gap(scene, xy, source_r)
        for obstacle in scene.objects.values():
            if obstacle.object_id == source.object_id:
                continue
            if obstacle.spec_name == "desk":
                gap = desk_gap
            else:
                gap = float(np.linalg.norm(xy - obstacle.position[:2])) - source_r - _footprint_radius(obstacle.spec_name)
            min_gap = min(min_gap, gap)
        inner_guard = bool(xy[0] < WORKTABLE_RELATIVE_INNER_GUARD_X_M and -0.14 <= xy[1] <= 0.10)
        return min_gap, desk_gap, inner_guard

    nominal_gap, nominal_desk_gap, nominal_inner_guard = _relative_clearance(nominal)
    if (
        nominal_gap >= 0.008
        and nominal_desk_gap >= WORKTABLE_SMALL_DESK_MIN_GAP_M
        and not nominal_inner_guard
    ):
        T_goal = _pose_xy_on_surface(source.T_world_obj, source.spec_name, nominal, _scene_object_surface_z(scene))
        return T_goal, desk.object_id, []

    best_xy = nominal.astype(np.float32)
    best_score = -float("inf")
    for xy in candidates:
        min_gap, desk_gap, inner_guard = _relative_clearance(xy)
        dist_nominal = float(np.linalg.norm(xy - nominal))
        projection = float(np.dot(xy - target.position[:2], vec))
        score = 2.0 * min_gap - 2.2 * dist_nominal + 0.15 * projection
        if min_gap > 0.012:
            score += 1.0
        if projection < 0.0:
            score -= 1.0
        if dist_nominal > 0.12:
            score -= 3.0 * (dist_nominal - 0.12)
        if desk_gap < WORKTABLE_SMALL_DESK_MIN_GAP_M:
            score -= 100.0 + 10.0 * (WORKTABLE_SMALL_DESK_MIN_GAP_M - desk_gap)
        if inner_guard:
            score -= 5.0 + 20.0 * (WORKTABLE_RELATIVE_INNER_GUARD_X_M - float(xy[0]))
        if score > best_score:
            best_score = score
            best_xy = xy

    warnings: list[str] = []
    shift = float(np.linalg.norm(best_xy - nominal))
    max_shift = MAX_RELATIVE_CLEARANCE_SHIFT_M if strict_distance else 0.35
    if shift > max_shift:
        warnings.append(
            f"relative clearance adjustment rejected: would shift target {shift:.3f}m from the requested relation"
        )
        best_xy = nominal.astype(np.float32)
        shift = 0.0
    if shift > 0.025:
        warnings.append(f"relative target shifted {shift:.3f}m for clearance/reachability")
    T_goal = _pose_xy_on_surface(source.T_world_obj, source.spec_name, best_xy, _scene_object_surface_z(scene))
    return T_goal, desk.object_id, warnings


def _relative_pose_with_warnings(
    scene: SceneState,
    source: SceneObject,
    target: SceneObject,
    direction: str,
    distance_m: float,
    *,
    strict_distance: bool = False,
) -> tuple[np.ndarray, str, list[str]]:
    vec_xy = _table_direction_xy(direction)
    vec = np.array([float(vec_xy[0]), float(vec_xy[1]), 0.0], dtype=np.float32)
    if target.spec_name == "desk" or _object_is_on_small_desk(scene, target):
        T_goal, support_id = _small_desk_empty_pose(
            scene,
            source,
            anchor_xy=target.position[:2],
            direction_xy=vec[:2],
            prefer_near=True,
        )
        return T_goal, support_id, ["relative target constrained onto the small desk surface"]
    return _worktable_relative_pose(scene, source, target, direction, distance_m, strict_distance=strict_distance)


def _relative_pose(scene: SceneState, source: SceneObject, target: SceneObject, direction: str, distance_m: float) -> tuple[np.ndarray, str]:
    T_goal, support_id, _warnings = _relative_pose_with_warnings(scene, source, target, direction, distance_m)
    return T_goal, support_id


def _lean_against_pose(
    scene: SceneState,
    source: SceneObject,
    target: SceneObject,
    *,
    side: str = "right",
    lean_angle_deg: float = 30.0,
) -> tuple[np.ndarray, str, list[str]]:
    side_vec = _table_direction_xy(str(side))
    toward_target = -side_vec
    angle = math.radians(float(np.clip(lean_angle_deg, 8.0, 65.0)))
    long_axis = np.array(
        [
            toward_target[0] * math.sin(angle),
            toward_target[1] * math.sin(angle),
            math.cos(angle),
        ],
        dtype=np.float32,
    )
    T = _set_long_axis_world(source.T_world_obj, source.spec_name, long_axis, side_axis_xy=side_vec)
    target_half = _xy_half_extent_along(target.T_world_obj, target.spec_name, side_vec)
    source_half = _xy_half_extent_along(T, source.spec_name, side_vec)
    xy = np.asarray(target.position[:2], dtype=np.float32) + side_vec * (target_half + source_half + 0.004)
    T = _pose_xy_on_surface(T, source.spec_name, xy, _scene_object_surface_z(scene), margin=0.002)
    warnings = [
        f"lean_against generated side={side}, lean_angle_deg={float(lean_angle_deg):.1f}; "
        "candidate planner will validate feasibility"
    ]
    return T, _desk_object(scene).object_id, warnings


def _object_support_top_z(scene: SceneState, support: SceneObject) -> float:
    if support.spec_name == "desk":
        return _scene_object_surface_z(scene)
    try:
        _, top = _precise_world_z_bounds(support.T_world_obj, support.spec_name)
    except Exception:
        _, top = _world_z_bounds(support.T_world_obj, support.spec_name)
    return float(top)


def _on_object_pose(
    scene: SceneState,
    source: SceneObject,
    target: SceneObject,
    *,
    orientation_policy: str | None = None,
    anchor: str = "top_center",
) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    support_top_z = _object_support_top_z(scene, target)
    T = _apply_orientation_policy(source, orientation_policy)
    xy = np.asarray(target.position[:2], dtype=np.float32)
    T = _pose_xy_on_surface(T, source.spec_name, xy, support_top_z, margin=0.002)

    if target.spec_name != "desk":
        source_xy = _precise_projected_xy_extents(T, source.spec_name)
        target_xy = _precise_projected_xy_extents(target.T_world_obj, target.spec_name)
        source_span = float(np.max(source_xy))
        target_span = float(np.min(target_xy))
        source_area = float(max(source_xy[0] * source_xy[1], 1e-6))
        target_area = float(max(target_xy[0] * target_xy[1], 1e-6))
        if source_span > target_span * 1.15:
            warnings.append(
                f"support may be too narrow: source_xy={source_xy.round(4).tolist()} "
                f"target_xy={target_xy.round(4).tolist()}"
            )
        if source_area > target_area * 0.85:
            warnings.append(
                f"support area margin is small: source_area={source_area:.5f} target_area={target_area:.5f}"
            )
        if source.spec_name == "tennis":
            warnings.append("round object on top target may roll; prefer drop/inside target when possible")
    return T, warnings


def _scene_object_surface_z(scene: SceneState) -> float:
    bottoms = []
    for obj in scene.objects.values():
        if obj.spec_name == "desk":
            continue
        try:
            bottom, _ = _precise_world_z_bounds(obj.T_world_obj, obj.spec_name)
        except Exception:
            bottom, _ = _world_z_bounds(obj.T_world_obj, obj.spec_name)
        if -0.03 <= bottom <= 0.08:
            bottoms.append(float(bottom))
    if bottoms:
        return float(np.median(np.asarray(bottoms, dtype=np.float32)))
    return 0.0


def _pose_xy_on_surface(T_template: np.ndarray, spec_name: str, xy: np.ndarray, surface_z: float, margin: float = 0.001) -> np.ndarray:
    T = np.asarray(T_template, dtype=np.float32).reshape(4, 4).copy()
    xy_arr = np.asarray(xy, dtype=np.float32).reshape(2)
    T[0, 3] = float(xy_arr[0])
    T[1, 3] = float(xy_arr[1])
    T[2, 3] = 0.0
    try:
        bottom, _ = _precise_world_z_bounds(T, spec_name)
    except Exception:
        bottom, _ = _world_z_bounds(T, spec_name)
    T[2, 3] = float(surface_z) + float(margin) - float(bottom)
    return T.astype(np.float32)


def _buffer_pose(scene: SceneState, source: SceneObject) -> tuple[np.ndarray, str]:
    desk = _desk_object(scene)
    candidates = []
    for x, z in [item[1] for item in DESK_SLOT_LAYOUT_XZ] + [(-0.20, 0.0), (0.20, 0.0), (0.0, 0.14), (0.0, -0.14)]:
        local = np.eye(4, dtype=np.float32)
        local[:3, 3] = np.array([float(x), 0.08, float(z)], dtype=np.float32)
        world = (desk.T_world_obj @ local)[:3, 3]
        min_clear = float("inf")
        for obj in scene.objects.values():
            if obj.object_id == source.object_id or obj.spec_name == "desk":
                continue
            min_clear = min(min_clear, float(np.linalg.norm(world[:2] - obj.position[:2])))
        candidates.append((min_clear, world))
    candidates.sort(key=lambda item: item[0], reverse=True)
    pos = candidates[0][1]
    return _preserve_pose_at_position(source, pos, scene, desk.object_id), desk.object_id


def _exchange_buffer_pose(scene: SceneState, source: SceneObject, other: SceneObject) -> tuple[np.ndarray, str]:
    if not _object_is_on_small_desk(scene, source) and not _object_is_on_small_desk(scene, other):
        try:
            mid = (source.position[:2] + other.position[:2]) * 0.5
            return _small_desk_empty_pose(scene, source, anchor_xy=mid, prefer_near=False)
        except Exception:
            pass
    desk = _desk_object(scene)
    surface_z = _scene_object_surface_z(scene)
    source_xy = source.position[:2]
    other_xy = other.position[:2]
    radius = _footprint_radius(source.spec_name)

    candidate_xy: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()

    def _add_world_candidate(xy: np.ndarray) -> None:
        xy = np.asarray(xy, dtype=np.float32).reshape(2)
        key = (round(float(xy[0]), 4), round(float(xy[1]), 4))
        if key in seen:
            return
        seen.add(key)
        candidate_xy.append(xy)

    mid = (source_xy + other_xy) * 0.5
    tabletop_xy = [
        np.asarray(obj.position[:2], dtype=np.float32).reshape(2)
        for obj in scene.objects.values()
        if obj.spec_name != "desk"
    ]
    if tabletop_xy:
        all_xy = np.vstack(tabletop_xy)
        span_min = np.min(all_xy, axis=0) - np.array([0.08, 0.08], dtype=np.float32)
        span_max = np.max(all_xy, axis=0) + np.array([0.08, 0.08], dtype=np.float32)
    else:
        span_min = mid - np.array([0.16, 0.16], dtype=np.float32)
        span_max = mid + np.array([0.16, 0.16], dtype=np.float32)
    span_min = np.minimum(span_min, mid - np.array([0.10, 0.10], dtype=np.float32))
    span_max = np.maximum(span_max, mid + np.array([0.10, 0.10], dtype=np.float32))

    for x in np.linspace(float(span_min[0]), float(span_max[0]), 11):
        for y in np.linspace(float(span_min[1]), float(span_max[1]), 11):
            _add_world_candidate(np.array([x, y], dtype=np.float32))

    direction = other_xy - source_xy
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = (direction / norm).astype(np.float32)
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)
    base_dist = max(0.08, radius + _footprint_radius(other.spec_name) + 0.035)
    for scale in (1.0, 1.35, 1.7):
        _add_world_candidate(mid + perp * (base_dist * scale))
        _add_world_candidate(mid - perp * (base_dist * scale))
    _add_world_candidate(source_xy - direction * base_dist)
    _add_world_candidate(other_xy + direction * base_dist)

    blocked_xy = [
        (source_xy, radius, "source_original"),
        (other_xy, _footprint_radius(other.spec_name), "other_original"),
    ]

    best_score = -float("inf")
    best_xy = candidate_xy[0]
    desired_gap = max(0.018, min(0.04, radius * 0.55))
    for xy in candidate_xy:
        min_gap = float("inf")
        for obj in scene.objects.values():
            if obj.object_id == source.object_id or obj.spec_name == "desk":
                continue
            obstacle_gap = float(np.linalg.norm(xy - obj.position[:2])) - radius - _footprint_radius(obj.spec_name)
            min_gap = min(min_gap, obstacle_gap)
        for reserved_xy, reserved_radius, _name in blocked_xy:
            reserved_gap = float(np.linalg.norm(xy - reserved_xy)) - radius - float(reserved_radius)
            min_gap = min(min_gap, reserved_gap)
        dist_mid = float(np.linalg.norm(xy - mid))
        if min_gap >= desired_gap:
            score = 2.0 + 0.35 * min_gap - 0.75 * dist_mid
        else:
            score = min_gap - 0.20 * dist_mid
        if score > best_score:
            best_score = score
            best_xy = xy
    return _pose_xy_on_surface(source.T_world_obj, source.spec_name, best_xy, surface_z), desk.object_id


def _pose_goal_ref_obj(scene: SceneState, value: Any, *, role: str) -> SceneObject:
    if isinstance(value, dict):
        for key in ("object", "target", "ref", "object_id", "name"):
            if key in value:
                return _pose_goal_ref_obj(scene, value[key], role=role)
    resolved = resolve_object_ref(scene, str(value), role=role)
    return scene.get(resolved["object_id"])


def _pose_goal_position_xy(
    scene: SceneState,
    source: SceneObject,
    expr: Any,
    warnings: list[str],
) -> tuple[np.ndarray, str]:
    if isinstance(expr, (list, tuple, np.ndarray)):
        arr = np.asarray(expr, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise ValueError("pose_goal position array must contain at least x/y")
        return arr[:2].astype(np.float32), "absolute_xy"
    if isinstance(expr, str):
        obj = _pose_goal_ref_obj(scene, expr, role="pose_position_anchor")
        return obj.position[:2].astype(np.float32), f"object_center({obj.object_id})"
    if not isinstance(expr, dict):
        raise ValueError(f"pose_goal.position must be an object, string, or xy array, got {type(expr).__name__}")

    expr_type = str(expr.get("type", "object_center")).strip()
    if expr_type in {"absolute_xyz", "xyz", "world_xyz"}:
        raw = np.asarray(expr.get("xyz_m", expr.get("position_xyz_m", expr.get("value"))), dtype=np.float32).reshape(-1)
        if raw.size < 2:
            raise ValueError("absolute_xyz pose position must contain at least x/y")
        return raw[:2].astype(np.float32), "absolute_xyz"
    if expr_type in {"absolute_xy", "xy", "world_xy"}:
        raw = np.asarray(expr.get("xy_m", expr.get("position_xy_m", expr.get("value"))), dtype=np.float32).reshape(-1)
        if raw.size < 2:
            raise ValueError("absolute_xy pose position must contain x/y")
        return raw[:2].astype(np.float32), "absolute_xy"
    if expr_type in {"current", "current_pose", "source_current"}:
        return source.position[:2].astype(np.float32), "current_xy"
    if expr_type in {"object_center", "center", "object"}:
        obj = _pose_goal_ref_obj(scene, expr.get("object", expr.get("target", expr.get("ref", source.object_id))), role="pose_position_object")
        return obj.position[:2].astype(np.float32), f"object_center({obj.object_id})"
    if expr_type in {"midpoint", "between"}:
        refs = list(expr.get("objects") or [])
        if len(refs) >= 2:
            a = _pose_goal_ref_obj(scene, refs[0], role="pose_midpoint_a")
            b = _pose_goal_ref_obj(scene, refs[1], role="pose_midpoint_b")
        else:
            a = _pose_goal_ref_obj(scene, expr.get("object_a", expr.get("a")), role="pose_midpoint_a")
            b = _pose_goal_ref_obj(scene, expr.get("object_b", expr.get("b")), role="pose_midpoint_b")
        return ((a.position[:2] + b.position[:2]) * 0.5).astype(np.float32), f"midpoint({a.object_id},{b.object_id})"
    if expr_type in {"linear_interpolation", "lerp", "line"}:
        a = _pose_goal_ref_obj(scene, expr.get("from", expr.get("object_a", expr.get("a"))), role="pose_lerp_from")
        b = _pose_goal_ref_obj(scene, expr.get("to", expr.get("object_b", expr.get("b"))), role="pose_lerp_to")
        alpha = float(expr.get("alpha_from_to", expr.get("alpha", 0.5)))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        xy = ((1.0 - alpha) * a.position[:2] + alpha * b.position[:2]).astype(np.float32)
        return xy, f"lerp({a.object_id},{b.object_id},alpha={alpha:.3f})"
    if expr_type == "offset":
        anchor_expr = expr.get("anchor", expr.get("from", expr.get("object", source.object_id)))
        anchor_xy, anchor_desc = _pose_goal_position_xy(scene, source, anchor_expr, warnings)
        offset = np.zeros(2, dtype=np.float32)
        if "offset_m" in expr:
            raw_offset = np.asarray(expr.get("offset_m"), dtype=np.float32).reshape(-1)
            if raw_offset.size >= 2:
                offset[:] = raw_offset[:2]
        else:
            offset[0] = float(expr.get("dx_m", expr.get("x_m", 0.0)) or 0.0)
            offset[1] = float(expr.get("dy_m", expr.get("y_m", 0.0)) or 0.0)
        direction = str(expr.get("direction", "")).strip()
        if direction:
            offset += _table_direction_xy(direction) * float(expr.get("distance_m", 0.0) or 0.0)
        return (anchor_xy + offset).astype(np.float32), f"offset({anchor_desc})"
    if expr_type in {"nearest_free", "nearest_empty"}:
        anchor_xy, anchor_desc = _pose_goal_position_xy(scene, source, expr.get("anchor", expr.get("around", source.object_id)), warnings)
        surface = str(expr.get("surface", "worktable"))
        if surface == "small_desk":
            T_free, _support_id = _small_desk_empty_pose(scene, source, anchor_xy=anchor_xy, prefer_near=True)
        else:
            T_free, _support_id = _surface_empty_pose(scene, source, anchor_xy=anchor_xy, prefer_near=True)
        return T_free[:2, 3].astype(np.float32), f"nearest_free({anchor_desc})"
    if expr_type in {"extreme_empty", "empty_extreme", "farthest_empty", "leftmost_empty", "rightmost_empty"}:
        relation = str(expr.get("relation", "") or "").strip()
        if not relation:
            relation = {
                "farthest_empty": "farthest",
                "leftmost_empty": "leftmost",
                "rightmost_empty": "rightmost",
            }.get(expr_type, "nearest")
        reference_xy = None
        if expr.get("reference") and str(expr.get("reference")) not in {"robot_base", "robot", "surface_center", "table_center"}:
            reference_xy = _pose_goal_ref_obj(scene, expr.get("reference"), role="pose_extreme_reference").position[:2]
        T_free, _support_id, extreme_warnings = _empty_pose_by_extreme(
            scene,
            source,
            surface=str(expr.get("surface", "worktable")),
            relation=relation,
            reference_xy=reference_xy,
            reference_frame=str(expr.get("reference_frame", "table_xy")),
        )
        warnings.extend(extreme_warnings)
        return T_free[:2, 3].astype(np.float32), f"extreme_empty({relation})"
    raise ValueError(f"Unsupported pose_goal.position.type: {expr_type!r}")


def _apply_pose_goal_orientation(
    scene: SceneState,
    source: SceneObject,
    T_template: np.ndarray,
    orientation: Any,
    goal_xy: np.ndarray,
    warnings: list[str],
) -> np.ndarray:
    T = np.asarray(T_template, dtype=np.float32).reshape(4, 4).copy()
    if orientation is None:
        return T
    if isinstance(orientation, str):
        orientation = {"type": orientation}
    if not isinstance(orientation, dict):
        raise ValueError("pose_goal.orientation must be a string or object")
    orient_type = str(orientation.get("type", "keep_current")).strip()
    if orient_type in {"", "keep_current", "current"}:
        return T
    if orient_type in {"upright", "vertical"}:
        T[:3, :3] = _upright_rotation(source)
        return T
    if orient_type in {"lay_flat", "horizontal", "flat"}:
        T[:3, :3] = _horizontal_rotation(source)
        return T
    if orient_type in {"rotate_yaw", "yaw"}:
        yaw_deg = float(orientation.get("yaw_deg", orientation.get("deg", 0.0)) or 0.0)
        T[:3, :3] = (_axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), math.radians(yaw_deg)) @ T[:3, :3]).astype(np.float32)
        return T
    if orient_type in {"long_axis_toward", "tip_toward", "toward"}:
        target = _pose_goal_ref_obj(scene, orientation.get("target", orientation.get("object")), role="pose_orientation_target")
        T[:2, 3] = np.asarray(goal_xy, dtype=np.float32).reshape(2)
        T = _orient_long_axis_toward(T, source, target)
        semantic_axis = str(orientation.get("semantic_axis", "long_axis"))
        if semantic_axis == "tip":
            warnings.append(f"oriented semantic tip/long axis toward {target.object_id}")
        else:
            warnings.append(f"oriented long axis toward {target.object_id}")
        return T
    if orient_type in {"long_axis_parallel_to", "parallel_to_line"}:
        a = _pose_goal_ref_obj(scene, orientation.get("object_a", orientation.get("from", orientation.get("a"))), role="pose_orientation_line_a")
        b = _pose_goal_ref_obj(scene, orientation.get("object_b", orientation.get("to", orientation.get("b"))), role="pose_orientation_line_b")
        T = _set_long_axis_xy(T, source.spec_name, b.position[:2] - a.position[:2])
        warnings.append(f"oriented long axis parallel to line {a.object_id}->{b.object_id}")
        return T
    if orient_type in {"parallel_table_edge", "desk_edge"}:
        T = _orient_parallel_to_desk_edge(scene, T, source)
        warnings.append("oriented long axis parallel to desk edge")
        return T
    raise ValueError(f"Unsupported pose_goal.orientation.type: {orient_type!r}")


def _resolve_pose_goal(scene: SceneState, source: SceneObject, pose_goal: dict[str, Any]) -> tuple[np.ndarray, str, list[str], str, str]:
    if not isinstance(pose_goal, dict):
        raise ValueError("pose_goal must be an object")
    warnings: list[str] = []
    position_expr = pose_goal.get("position", pose_goal.get("target_position", pose_goal))
    orientation_expr = pose_goal.get("orientation", None)
    position_type = str(position_expr.get("type", "") if isinstance(position_expr, dict) else "").strip()
    support_id = _desk_object(scene).object_id
    place_mode = "surface_place"

    if position_type == "slot":
        T_goal, support_id = _slot_pose(scene, source, str(position_expr.get("slot", "slot_1")))
        T_goal = _apply_pose_goal_orientation(scene, source, T_goal, orientation_expr, T_goal[:2, 3], warnings)
        T_goal = _pose_xy_on_surface(T_goal, source.spec_name, T_goal[:2, 3], _desk_slot_support_z(_desk_object(scene)))
        return T_goal, support_id, warnings, "surface_place", "pose_goal slot"
    if position_type in {"inside", "in"}:
        target = _pose_goal_ref_obj(scene, position_expr.get("target", position_expr.get("object")), role="pose_inside_target")
        T_goal, inside_warnings = _inside_pose(scene, source, target)
        warnings.extend(inside_warnings)
        return T_goal, target.object_id, warnings, "drop_place", f"pose_goal inside {target.object_id}"
    if position_type in {"on_top", "on"}:
        target = _pose_goal_ref_obj(scene, position_expr.get("target", position_expr.get("object")), role="pose_on_target")
        support_z = _object_support_top_z(scene, target)
        T_template = _apply_pose_goal_orientation(scene, source, source.T_world_obj, orientation_expr, target.position[:2], warnings)
        T_goal = _pose_xy_on_surface(T_template, source.spec_name, target.position[:2], support_z, margin=0.002)
        return T_goal, target.object_id, warnings, "surface_place", f"pose_goal on_top {target.object_id}"
    if position_type == "offset" and isinstance(position_expr, dict):
        anchor_value = position_expr.get("anchor", position_expr.get("from", position_expr.get("object")))
        if anchor_value is not None:
            try:
                anchor_obj = _pose_goal_ref_obj(scene, anchor_value, role="pose_offset_anchor")
                explicit_surface = position_expr.get("surface", pose_goal.get("surface"))
                if explicit_surface is None and _object_is_on_small_desk(scene, anchor_obj):
                    raw_offset = np.zeros(2, dtype=np.float32)
                    if "offset_m" in position_expr:
                        arr = np.asarray(position_expr.get("offset_m"), dtype=np.float32).reshape(-1)
                        if arr.size >= 2:
                            raw_offset[:] = arr[:2]
                    direction = str(position_expr.get("direction", "")).strip()
                    if direction:
                        raw_offset += _table_direction_xy(direction) * float(position_expr.get("distance_m", 0.0) or 0.0)
                    T_goal, support_id = _small_desk_empty_pose(
                        scene,
                        source,
                        anchor_xy=anchor_obj.position[:2],
                        direction_xy=raw_offset if float(np.linalg.norm(raw_offset)) > 1e-6 else None,
                        prefer_near=True,
                    )
                    T_goal = _apply_pose_goal_orientation(scene, source, T_goal, orientation_expr, T_goal[:2, 3], warnings)
                    warnings.append(f"offset around {anchor_obj.object_id} constrained onto small desk surface")
                    return T_goal, support_id, warnings, "surface_place", f"pose_goal small_desk_offset({anchor_obj.object_id})"
            except Exception as exc:
                warnings.append(f"ignored small-desk offset constraint: {exc!r}")

    xy, desc = _pose_goal_position_xy(scene, source, position_expr, warnings)
    surface = str((position_expr.get("surface") if isinstance(position_expr, dict) else None) or pose_goal.get("surface", "worktable"))
    should_clearance_adjust = position_type in {"midpoint", "between"}
    if position_type == "offset" and isinstance(position_expr, dict):
        offset_norm = 0.0
        if "offset_m" in position_expr:
            raw_offset = np.asarray(position_expr.get("offset_m"), dtype=np.float32).reshape(-1)
            if raw_offset.size >= 2:
                offset_norm = float(np.linalg.norm(raw_offset[:2]))
        else:
            offset_norm = abs(float(position_expr.get("distance_m", 0.0) or 0.0))
        should_clearance_adjust = bool(offset_norm <= 0.075)
    if should_clearance_adjust:
        xy, clearance_warnings = _pose_goal_clearance_adjusted_xy(scene, source, xy, surface=surface)
        warnings.extend(clearance_warnings)
    T_template = _apply_pose_goal_orientation(scene, source, source.T_world_obj, orientation_expr, xy, warnings)
    if surface == "small_desk":
        support_id = _desk_object(scene).object_id
        surface_z = _desk_slot_support_z(_desk_object(scene))
    else:
        surface_z = _scene_object_surface_z(scene)
    T_goal = _pose_xy_on_surface(T_template, source.spec_name, xy, surface_z, margin=0.002)
    if isinstance(position_expr, dict) and position_type in {"absolute_xyz", "xyz", "world_xyz"}:
        z_policy = str(position_expr.get("z_policy", "surface") or "surface").strip()
        if z_policy in {"exact", "exact_center", "center_z"}:
            raw_xyz = np.asarray(position_expr.get("xyz_m", position_expr.get("position_xyz_m")), dtype=np.float32).reshape(-1)
            if raw_xyz.size >= 3:
                T_goal[2, 3] = float(raw_xyz[2])
                warnings.append("used exact LLM-provided target center z")
    return T_goal, support_id, warnings, place_mode, f"pose_goal {desc}"


def _resolve_single_step(scene: SceneState, llm_step: dict[str, Any], index: int) -> ResolvedStep:
    op = str(llm_step["operator"])
    source_ref = resolve_object_ref(scene, str(llm_step["source_ref"]), role="source")
    source = scene.get(source_ref["object_id"])
    warnings: list[str] = []
    primitive = "place_on_slots"
    place_mode = "surface_place"
    support_id = _desk_object(scene).object_id

    if op == "inside":
        target_ref = resolve_object_ref(scene, str(llm_step["target_ref"]), role="destination")
        target = scene.get(target_ref["object_id"])
        T_goal, extra_warnings = _inside_pose(scene, source, target)
        warnings.extend(extra_warnings)
        support_id = target.object_id
        if source.spec_name == "bi" and target.spec_name == "bitong":
            primitive = "insert_vertical"
            place_mode = "auto"
        else:
            primitive = "place_on_slots"
            place_mode = "drop_place"
        desc = f"{source.object_id} -> inside {target.object_id}"
    elif op == "pose_goal":
        T_goal, support_id, pose_warnings, pose_place_mode, pose_desc = _resolve_pose_goal(
            scene,
            source,
            dict(llm_step.get("pose_goal") or llm_step.get("target_pose") or {}),
        )
        warnings.extend(pose_warnings)
        place_mode = pose_place_mode
        desc = f"{source.object_id} -> {pose_desc}"
    elif op == "between":
        a_ref = resolve_object_ref(scene, str(llm_step["object_a_ref"]), role="between_a")
        b_ref = resolve_object_ref(scene, str(llm_step["object_b_ref"]), role="between_b")
        a = scene.get(a_ref["object_id"])
        b = scene.get(b_ref["object_id"])
        T_goal, support_id, clearance_warnings = _between_clearance_pose(scene, source, a, b)
        warnings.extend(clearance_warnings)
        if llm_step.get("face_ref"):
            face_ref = resolve_object_ref(scene, str(llm_step["face_ref"]), role="face_target")
            face_target = scene.get(face_ref["object_id"])
            T_goal = _orient_long_axis_toward(T_goal, source, face_target)
            warnings.append(f"oriented long axis toward {face_target.object_id}")
        desc = f"{source.object_id} -> between {a.object_id} and {b.object_id}"
    elif op == "on":
        target_ref = resolve_object_ref(scene, str(llm_step["target_ref"]), role="destination")
        target = scene.get(target_ref["object_id"])
        support_id = target.object_id
        T_goal, extra_warnings = _on_object_pose(
            scene,
            source,
            target,
            orientation_policy=llm_step.get("orientation_policy"),
            anchor=str(llm_step.get("anchor", "top_center")),
        )
        warnings.extend(extra_warnings)
        desc = f"{source.object_id} -> on {target.object_id}"
    elif op == "relative":
        target_ref = resolve_object_ref(scene, str(llm_step["target_ref"]), role="destination")
        target = scene.get(target_ref["object_id"])
        T_goal, support_id, relative_warnings = _relative_pose_with_warnings(
            scene,
            source,
            target,
            str(llm_step.get("direction", "right")),
            float(llm_step.get("distance_m", 0.08)),
            strict_distance=bool(llm_step.get("strict_distance", False)),
        )
        warnings.extend(relative_warnings)
        if llm_step.get("parallel_to") == "desk_edge":
            T_goal = _orient_parallel_to_desk_edge(scene, T_goal, source)
            warnings.append("oriented long axis parallel to desk edge")
        elif llm_step.get("long_axis") == "toward_target":
            T_goal = _orient_long_axis_toward(T_goal, source, target)
            warnings.append(f"oriented long axis toward {target.object_id}")
        desc = f"{source.object_id} -> {llm_step.get('direction')} of {target.object_id}"
    elif op == "lean_against":
        target_ref = resolve_object_ref(scene, str(llm_step["target_ref"]), role="lean_target")
        target = scene.get(target_ref["object_id"])
        T_goal, support_id, lean_warnings = _lean_against_pose(
            scene,
            source,
            target,
            side=str(llm_step.get("side", "right")),
            lean_angle_deg=float(llm_step.get("lean_angle_deg", 30.0)),
        )
        warnings.extend(lean_warnings)
        desc = f"{source.object_id} -> lean against {target.object_id} {llm_step.get('side', 'right')}"
    elif op == "nearest_empty":
        target_ref = resolve_object_ref(scene, str(llm_step["target_ref"]), role="empty_anchor")
        target = scene.get(target_ref["object_id"])
        if str(llm_step.get("surface", "")) == "small_desk" or target.spec_name == "desk":
            T_goal, support_id = _small_desk_empty_pose(
                scene,
                source,
                anchor_xy=target.position[:2],
                prefer_near=True,
            )
        else:
            T_goal, support_id = _surface_empty_pose(scene, source, anchor_xy=target.position[:2], prefer_near=True)
        desc = f"{source.object_id} -> nearest empty space around {target.object_id}"
    elif op == "extreme_empty":
        reference_xy = None
        ref_value = llm_step.get("reference")
        if ref_value and str(ref_value) not in {"robot", "robot_base", "surface_center", "table_center"}:
            try:
                ref_obj = scene.get(resolve_object_ref(scene, str(ref_value), role="empty_reference")["object_id"])
                reference_xy = ref_obj.position[:2]
            except Exception as exc:
                warnings.append(f"ignored unresolved empty reference {ref_value!r}: {exc}")
        surface = str(llm_step.get("surface", "worktable"))
        if surface != "small_desk":
            try:
                target_ref = resolve_object_ref(scene, str(llm_step.get("target_ref", "")), role="empty_anchor")
                target = scene.get(target_ref["object_id"])
                if target.spec_name == "desk":
                    surface = "small_desk"
            except Exception:
                pass
        T_goal, support_id, extreme_warnings = _empty_pose_by_extreme(
            scene,
            source,
            surface=surface,
            relation=str(llm_step.get("relation", "nearest")),
            reference_xy=reference_xy,
            reference_frame=str(llm_step.get("reference_frame", "table_xy")),
        )
        warnings.extend(extreme_warnings)
        desc = f"{source.object_id} -> {llm_step.get('relation', 'nearest')} empty space ({surface})"
    elif op == "on_or_side_fallback":
        target_ref = resolve_object_ref(scene, str(llm_step["target_ref"]), role="destination")
        target = scene.get(target_ref["object_id"])
        on_goal, extra_warnings = _on_object_pose(scene, source, target)
        unstable = any("support" in warning or "roll" in warning for warning in extra_warnings)
        if unstable:
            T_goal, support_id, relative_warnings = _relative_pose_with_warnings(
                scene,
                source,
                target,
                str(llm_step.get("fallback_direction", "right")),
                float(llm_step.get("fallback_distance_m", 0.02)),
            )
            warnings.extend(extra_warnings)
            warnings.extend(relative_warnings)
            if llm_step.get("fallback_long_axis") == "parallel_table_edge":
                T_goal = _orient_parallel_to_desk_edge(scene, T_goal, source)
                warnings.append("oriented fallback long axis parallel to desk edge")
            elif llm_step.get("fallback_long_axis") == "toward_target":
                T_goal = _orient_long_axis_toward(T_goal, source, target)
                warnings.append(f"oriented fallback long axis toward {target.object_id}")
            warnings.append(f"fallback selected because support check was unstable for {target.object_id}")
            desc = f"{source.object_id} -> side of {target.object_id} after stability fallback"
        else:
            support_id = target.object_id
            T_goal = on_goal
            warnings.extend(extra_warnings)
            desc = f"{source.object_id} -> on {target.object_id}"
    elif op == "move_blocking_object":
        src_ref = resolve_object_ref(scene, str(llm_step["source_ref"]), role="blocked_source")
        dst_ref = resolve_object_ref(scene, str(llm_step["target_ref"]), role="blocked_target")
        blocked_source = scene.get(src_ref["object_id"])
        blocked_target = scene.get(dst_ref["object_id"])
        blocker = _blocking_object_for(scene, blocked_source, blocked_target)
        if blocker is None:
            raise ValueError("No blocking object candidate found")
        source = blocker
        T_goal, support_id = _surface_empty_pose(
            scene,
            source,
            anchor_xy=blocker.position[:2],
            prefer_near=False,
            right_half=True,
        )
        warnings.append(f"selected likely blocker {blocker.object_id} for clear-path pre-step")
        desc = f"{source.object_id} -> clear path for {blocked_source.object_id}"
    elif op == "rotate_in_place":
        yaw_deg = float(llm_step.get("yaw_deg", 0.0))
        T_goal = np.asarray(source.T_world_obj, dtype=np.float32).reshape(4, 4).copy()
        T_goal[:3, :3] = (_axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), math.radians(yaw_deg)) @ T_goal[:3, :3]).astype(np.float32)
        T_goal = _pose_xy_on_surface(T_goal, source.spec_name, source.position[:2], _scene_object_surface_z(scene))
        desc = f"{source.object_id} rotate yaw {yaw_deg:g}deg in place"
    elif op == "upright_in_place":
        T_goal = np.asarray(source.T_world_obj, dtype=np.float32).reshape(4, 4).copy()
        T_goal[:3, :3] = _upright_rotation(source)
        T_goal = _pose_xy_on_surface(T_goal, source.spec_name, source.position[:2], _scene_object_surface_z(scene))
        place_mode = "vertical_place"
        desc = f"{source.object_id} upright in place"
    elif op == "desk_slot":
        T_goal, support_id = _slot_pose(scene, source, str(llm_step["slot_name"]))
        desc = f"{source.object_id} -> {llm_step['slot_name']}"
    else:
        raise ValueError(f"Unsupported operator: {op}")

    return ResolvedStep(
        index=index,
        source_id=source.object_id,
        operator=op,
        target_object_id=support_id,
        target_pose=T_goal,
        place_mode=place_mode,
        primitive=primitive,
        description=desc,
        mock_llm_step=llm_step,
        warnings=warnings,
    )


def resolve_plan(scene: SceneState, llm_plan: dict[str, Any]) -> list[ResolvedStep]:
    if llm_plan.get("operator") == "sequence":
        working_scene = scene.copy()
        resolved: list[ResolvedStep] = []
        sub_plans = _reorder_dependent_support_moves(scene, list(llm_plan.get("steps", []) or []))
        for sub_plan in sub_plans:
            sub_steps = resolve_plan(working_scene, sub_plan)
            for step in sub_steps:
                if (
                    step.operator == "rotate_in_place"
                    and resolved
                    and resolved[-1].source_id == step.source_id
                ):
                    previous = resolved[-1]
                    previous.target_pose = np.asarray(step.target_pose, dtype=np.float32).reshape(4, 4).copy()
                    if working_scene.get(previous.source_id).spec_name == "hongshupian":
                        previous.target_pose[2, 3] += 0.006
                        previous.place_mode = "drop_place"
                        previous.warnings.append(
                            "raised merged hongshupian rotate target by 6mm and used drop_place to avoid final-contact table collision"
                        )
                    previous.description = f"{previous.description}; then rotate in place"
                    previous.mock_llm_step = {
                        "operator": "merged_place_then_rotate_in_place",
                        "steps": [previous.mock_llm_step, step.mock_llm_step],
                    }
                    previous.warnings.extend(step.warnings)
                    previous.warnings.append("merged consecutive rotate_in_place into the prior placement target")
                    working_scene.update_pose(previous.source_id, previous.target_pose, placed=True)
                    continue
                step.index = len(resolved) + 1
                resolved.append(step)
                working_scene.update_pose(step.source_id, step.target_pose, placed=True)
        return resolved

    if llm_plan.get("operator") == "collection_slots":
        working_scene = scene.copy()
        resolved: list[ResolvedStep] = []
        source_refs = list(llm_plan.get("source_refs", []) or [])
        slot_names = list(llm_plan.get("slot_names", []) or [])
        for source_ref, slot_name in zip(source_refs, slot_names):
            sub_plan = {
                "kind": "single_step",
                "operator": "desk_slot",
                "source_ref": str(source_ref),
                "slot_name": str(slot_name),
                "notes": "expanded from collection_slots",
            }
            step = _resolve_single_step(working_scene, sub_plan, len(resolved) + 1)
            step.operator = "collection_slot"
            resolved.append(step)
            working_scene.update_pose(step.source_id, step.target_pose, placed=True)
        return resolved

    if llm_plan.get("operator") == "collection_right_half":
        working_scene = scene.copy()
        resolved: list[ResolvedStep] = []
        source_refs = list(llm_plan.get("source_refs", []) or [])
        movable = [working_scene.get(str(ref)) for ref in source_refs if str(ref) in working_scene.objects]
        if movable:
            target_surface = str(llm_plan.get("target_surface", "worktable"))
            if target_surface == "small_desk":
                preferred_slots = {
                    "hongshupian": "slot_1",
                    "carriot": "slot_2",
                    "shuazi": "slot_3",
                    "gluestick": "slot_4",
                    "bi": "slot_5",
                    "lvmukuai": "slot_6",
                }
                fallback_slots = [f"slot_{idx}" for idx in range(1, 7)]
                if len(movable) > len(fallback_slots):
                    preferred_movable = [obj for obj in movable if obj.spec_name in preferred_slots]
                    extra_movable = [obj for obj in movable if obj.spec_name not in preferred_slots]
                    movable = (preferred_movable + extra_movable)[: len(fallback_slots)]
                used_slots: set[str] = set()
                for obj in movable:
                    preferred = preferred_slots.get(obj.spec_name)
                    if preferred is not None and preferred not in used_slots:
                        slot_name = preferred
                    else:
                        slot_name = next((name for name in fallback_slots if name not in used_slots), fallback_slots[-1])
                    used_slots.add(slot_name)
                    T_goal, _support_id = _slot_pose(working_scene, obj, slot_name)
                    step = ResolvedStep(
                        index=len(resolved) + 1,
                        source_id=obj.object_id,
                        operator="collection_right_half",
                        target_object_id=_desk_object(working_scene).object_id,
                        target_pose=T_goal,
                        place_mode="surface_place",
                        primitive="place_on_slots",
                        description=f"{obj.object_id} -> small-desk right-half cleanup {slot_name}",
                        mock_llm_step=llm_plan,
                        warnings=["expanded from cleanup command onto the small desk right half"],
                    )
                    resolved.append(step)
                    working_scene.update_pose(step.source_id, step.target_pose, placed=True)
            else:
                object_xy = np.vstack([obj.position[:2] for obj in working_scene.objects.values() if obj.spec_name != "desk"])
                center_x = float(np.mean(object_xy[:, 0]))
                y_min = float(np.min(object_xy[:, 1]) - 0.08)
                y_max = float(np.max(object_xy[:, 1]) + 0.08)
                search_min_x = center_x + 0.035
                search_max_x = center_x + 0.235
                x_values = np.linspace(search_min_x, search_max_x, 3)
                y_values = np.linspace(y_min, y_max, max(1, math.ceil(len(movable) / 3)))
                anchors = [np.array([x, y], dtype=np.float32) for y in y_values for x in x_values]
                for obj, anchor_xy in zip(movable, anchors):
                    source_r = _footprint_radius(obj.spec_name)
                    best_xy = np.asarray(anchor_xy, dtype=np.float32).reshape(2)
                    best_score = -float("inf")
                    for x in np.linspace(search_min_x, search_max_x, 9):
                        for y in np.linspace(y_min, y_max, 13):
                            xy = np.array([x, y], dtype=np.float32)
                            min_gap = float("inf")
                            desk_gap = _desk_xy_gap(working_scene, xy, source_r)
                            for obstacle in working_scene.objects.values():
                                if obstacle.object_id == obj.object_id:
                                    continue
                                if obstacle.spec_name == "desk":
                                    gap = desk_gap
                                else:
                                    gap = (
                                        float(np.linalg.norm(xy - obstacle.position[:2]))
                                        - source_r
                                        - _footprint_radius(obstacle.spec_name)
                                    )
                                min_gap = min(min_gap, gap)
                            dist_anchor = float(np.linalg.norm(xy - anchor_xy))
                            score = 3.0 * min_gap - 0.35 * dist_anchor + 0.05 * float(xy[0])
                            if min_gap > 0.015:
                                score += 1.0
                            if desk_gap < WORKTABLE_SMALL_DESK_MIN_GAP_M:
                                score -= 100.0 + 10.0 * (WORKTABLE_SMALL_DESK_MIN_GAP_M - desk_gap)
                            if score > best_score:
                                best_score = score
                                best_xy = xy
                    T_goal = _pose_xy_on_surface(
                        _worktable_surface_template(working_scene, obj),
                        obj.spec_name,
                        best_xy,
                        _scene_object_surface_z(working_scene),
                    )
                    warnings = ["expanded from cleanup command onto the worktable right half"]
                    if obj.spec_name == "gluestick":
                        warnings.append("worktable cleanup keeps gluestick horizontal instead of slot-upright")
                    desk_gap = _desk_xy_gap(working_scene, best_xy, source_r)
                    if desk_gap < WORKTABLE_SMALL_DESK_MIN_GAP_M:
                        warnings.append(f"worktable target is close to small desk: gap={desk_gap:.3f}m")
                    step = ResolvedStep(
                        index=len(resolved) + 1,
                        source_id=obj.object_id,
                        operator="collection_right_half",
                        target_object_id=_desk_object(working_scene).object_id,
                        target_pose=T_goal,
                        place_mode="surface_place",
                        primitive="place_on_slots",
                        description=f"{obj.object_id} -> worktable right-half cleanup",
                        mock_llm_step=llm_plan,
                        warnings=warnings,
                    )
                    resolved.append(step)
                    working_scene.update_pose(step.source_id, step.target_pose, placed=True)
        return resolved

    if llm_plan.get("operator") == "exchange":
        a_ref = resolve_object_ref(scene, str(llm_plan["source_ref"]), role="exchange_a")
        b_ref = resolve_object_ref(scene, str(llm_plan["other_ref"]), role="exchange_b")
        a = scene.get(a_ref["object_id"])
        b = scene.get(b_ref["object_id"])
        buffer_obj = a
        other_obj = b
        if a.spec_name == "tennis" and b.spec_name != "tennis":
            buffer_obj = b
            other_obj = a
        buffer_initial = buffer_obj.T_world_obj.copy()
        other_initial = other_obj.T_world_obj.copy()
        support_id = _desk_object(scene).object_id
        surface_z = _scene_object_surface_z(scene)
        other_at_buffer_pose = _pose_xy_on_surface(
            other_initial,
            other_obj.spec_name,
            buffer_initial[:2, 3],
            surface_z,
        )
        buffer_at_other_pose = _pose_xy_on_surface(
            buffer_initial,
            buffer_obj.spec_name,
            other_initial[:2, 3],
            surface_z,
        )
        buffer_T, support_id = _exchange_buffer_pose(scene, buffer_obj, other_obj)
        return [
            ResolvedStep(
                index=1,
                source_id=buffer_obj.object_id,
                operator="buffer",
                target_object_id=support_id,
                target_pose=buffer_T,
                place_mode="surface_place",
                primitive="place_on_slots",
                description=f"{buffer_obj.object_id} -> temporary buffer",
                mock_llm_step={"operator": "buffer", "source_ref": buffer_obj.object_id},
                warnings=[],
            ),
            ResolvedStep(
                index=2,
                source_id=other_obj.object_id,
                operator="exchange_to_a_pose",
                target_object_id=support_id,
                target_pose=other_at_buffer_pose,
                place_mode="surface_place",
                primitive="place_on_slots",
                description=f"{other_obj.object_id} -> original pose of {buffer_obj.object_id}",
                mock_llm_step={"operator": "exchange_to_a_pose", "source_ref": other_obj.object_id},
                warnings=[],
            ),
            ResolvedStep(
                index=3,
                source_id=buffer_obj.object_id,
                operator="exchange_to_b_pose",
                target_object_id=support_id,
                target_pose=buffer_at_other_pose,
                place_mode="surface_place",
                primitive="place_on_slots",
                description=f"{buffer_obj.object_id} -> original pose of {other_obj.object_id}",
                mock_llm_step={"operator": "exchange_to_b_pose", "source_ref": buffer_obj.object_id},
                warnings=[],
            ),
        ]
    return [_resolve_single_step(scene, llm_plan, 1)]


def _dynamic_rule_for_step(scene: SceneState, step: ResolvedStep) -> dict[str, Any]:
    source = scene.get(step.source_id)
    target = scene.get(step.target_object_id)
    if step.primitive == "insert_vertical" and source.spec_name == "bi" and target.spec_name == "bitong":
        default_rule = get_place_rule("bi")
        if default_rule is not None and default_rule.object_pose_local is not None:
            return {
                "source_object_name": source.spec_name,
                "target_object_name": target.spec_name,
                "primitive": "insert_vertical",
                "hover_height": 0.15,
                "release_retreat_height": 0.15,
                "allow_long_axis_flip": True,
                "object_pose_local": {
                    "position": list(default_rule.object_pose_local.position),
                    "rpy_deg": list(default_rule.object_pose_local.rpy_deg),
                },
            }
    T_local = np.linalg.inv(target.T_world_obj) @ np.asarray(step.target_pose, dtype=np.float32).reshape(4, 4)
    source_rule = get_place_rule(source.spec_name)
    rule = {
        "source_object_name": source.spec_name,
        "target_object_name": target.spec_name,
        "primitive": "place_on_slots",
        "hover_height": 0.12 if step.place_mode == "vertical_place" else 0.09,
        "release_retreat_height": 0.12 if step.place_mode == "vertical_place" else 0.10,
        "preserve_long_axis_vertical": bool(step.place_mode == "vertical_place"),
        "orientation_invariant": bool(source.spec_name == "tennis"),
        "allow_tabletop_yaw_variants": bool(source.spec_name in {"tennis", "gluestick", "hongshupian", "bi"}),
        "slots": [
            {
                "name": f"llm_goal_{int(step.index):02d}",
                "object_pose_local": _local_pose_from_matrix(T_local),
            }
        ],
    }
    if source_rule is not None:
        rule["face_robot_axis_local"] = getattr(source_rule, "face_robot_axis_local", None)
        rule["tabletop_place_tcp_verticality_target"] = getattr(source_rule, "tabletop_place_tcp_verticality_target", None)
        rule["tabletop_place_tcp_axis_vertical"] = getattr(source_rule, "tabletop_place_tcp_axis_vertical", None)
    return {k: v for k, v in rule.items() if v is not None}


def _rough_overlap_warnings(scene: SceneState, step: ResolvedStep) -> list[str]:
    source = scene.get(step.source_id)
    p = np.asarray(step.target_pose[:3, 3], dtype=np.float32)
    source_r = _footprint_radius(source.spec_name)
    warnings = []
    for obj in scene.objects.values():
        if obj.object_id in {step.source_id, step.target_object_id} or obj.spec_name == "desk":
            continue
        dist = float(np.linalg.norm(p[:2] - obj.position[:2]))
        threshold = 0.65 * (source_r + _footprint_radius(obj.spec_name))
        if dist < threshold:
            warnings.append(
                f"rough footprint overlap risk: target {source.object_id} vs {obj.object_id}, xy_dist={dist:.3f}m"
            )
    return warnings


def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] <= 2:
        return pts
    unique = sorted({(float(x), float(y)) for x, y in pts})
    if len(unique) <= 2:
        return np.asarray(unique, dtype=np.float32)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def _footprint_hull(T_world_obj: np.ndarray, spec_name: str) -> np.ndarray:
    return _convex_hull_xy(_world_points(T_world_obj, spec_name)[:, :2])


def _preview_color(name: str, alpha: int = 190) -> tuple[int, int, int, int]:
    palette = {
        "lvmukuai": (34, 139, 86),
        "carriot": (221, 105, 35),
        "shuazi": (42, 110, 185),
        "hongshupian": (196, 48, 61),
        "gluestick": (92, 80, 190),
        "bi": (180, 38, 45),
        "tennis": (216, 188, 45),
        "bitong": (74, 98, 122),
        "desk": (129, 101, 73),
    }
    rgb = palette.get(str(name), (86, 106, 132))
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(alpha))


def _draw_label(draw, xy: tuple[float, float], text: str, fill=(15, 23, 42, 255), bg=(255, 255, 255, 218)) -> None:
    text = str(text)
    try:
        bbox = draw.textbbox((0, 0), text)
        tw = int(bbox[2] - bbox[0])
        th = int(bbox[3] - bbox[1])
    except Exception:
        tw = 7 * len(text)
        th = 12
    x, y = float(xy[0]), float(xy[1])
    pad = 3
    draw.rounded_rectangle((x, y, x + tw + 2 * pad, y + th + 2 * pad), radius=3, fill=bg)
    draw.text((x + pad, y + pad), text, fill=fill)


def _scene_robot_base_world_xyz_with_source(scene: SceneState) -> tuple[np.ndarray, str]:
    for key in (
        "robot_base_world_xyz_m",
        "robot_base_world_p",
        "robot_base_world_position",
        "robot_base_position",
    ):
        value = scene.raw_data.get(key)
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                return arr[:3].astype(np.float32), key
        except Exception:
            pass
    value = scene.raw_data.get("robot_base_world_pose")
    if isinstance(value, dict):
        for key in ("position", "p", "xyz"):
            try:
                arr = np.asarray(value.get(key), dtype=np.float32).reshape(-1)
                if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                    return arr[:3].astype(np.float32), f"robot_base_world_pose.{key}"
            except Exception:
                pass
    return DEFAULT_ROBOT_BASE_WORLD_XYZ.copy(), "fallback"


def _scene_robot_base_world_xyz(scene: SceneState) -> np.ndarray:
    return _scene_robot_base_world_xyz_with_source(scene)[0]


def _draw_robot_base_2d(draw, to_px, base_xyz: np.ndarray, *, draw_reach: bool = True, label: str = "robot base") -> None:
    base_xy = np.asarray(base_xyz, dtype=np.float32).reshape(3)[:2]
    if draw_reach:
        for radius, color in ((0.25, (148, 163, 184, 135)), (0.45, (148, 163, 184, 95)), (0.65, (148, 163, 184, 65))):
            x0, y0 = to_px(base_xy + [-radius, -radius])
            x1, y1 = to_px(base_xy + [radius, radius])
            draw.ellipse((x0, y1, x1, y0), outline=color, width=2)
    bx, by = to_px(base_xy)
    r_px = max(9.0, min(abs(to_px(base_xy + [0.035, 0.0])[0] - bx), abs(to_px(base_xy + [0.0, 0.035])[1] - by)))
    draw.ellipse((bx - r_px, by - r_px, bx + r_px, by + r_px), fill=(15, 23, 42, 230), outline=(255, 255, 255, 230), width=2)
    tip = to_px(base_xy + [0.085, 0.0])
    draw.line((bx, by, tip[0], tip[1]), fill=(15, 23, 42, 230), width=4)
    draw.polygon([(tip[0], tip[1]), (tip[0] - 10, tip[1] - 5), (tip[0] - 10, tip[1] + 5)], fill=(15, 23, 42, 230))
    _draw_label(draw, (bx + 10, by + 10), label, fill=(255, 255, 255, 255), bg=(15, 23, 42, 220))


def _draw_table_frame_axes_2d(draw, to_px, min_xy: np.ndarray, max_xy: np.ndarray) -> None:
    span = np.maximum(np.asarray(max_xy, dtype=np.float32) - np.asarray(min_xy, dtype=np.float32), 1e-6)
    origin = np.asarray([float(min_xy[0] + span[0] * 0.055), float(min_xy[1] + span[1] * 0.080)], dtype=np.float32)
    axis_len = float(min(max(span[0] * 0.18, 0.06), 0.14))
    ox, oy = to_px(origin)
    x_tip = to_px(origin + np.asarray([axis_len, 0.0], dtype=np.float32))
    y_tip = to_px(origin + np.asarray([0.0, axis_len], dtype=np.float32))
    draw.line((ox, oy, x_tip[0], x_tip[1]), fill=(220, 38, 38, 230), width=4)
    draw.line((ox, oy, y_tip[0], y_tip[1]), fill=(22, 163, 74, 230), width=4)
    draw.ellipse((ox - 4, oy - 4, ox + 4, oy + 4), fill=(15, 23, 42, 220))
    _draw_label(draw, (x_tip[0] + 5, x_tip[1] - 10), "+X front", fill=(127, 29, 29, 255), bg=(254, 226, 226, 230))
    _draw_label(draw, (y_tip[0] + 5, y_tip[1] - 10), "+Y left", fill=(20, 83, 45, 255), bg=(220, 252, 231, 230))


def _cuboid_faces(points: np.ndarray) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(8, 3)
    return [
        pts[[0, 1, 3, 2]],
        pts[[4, 5, 7, 6]],
        pts[[0, 1, 5, 4]],
        pts[[2, 3, 7, 6]],
        pts[[0, 2, 6, 4]],
        pts[[1, 3, 7, 5]],
    ]


def _mpl_rgba(name: str, alpha: float) -> tuple[float, float, float, float]:
    r, g, b, _ = _preview_color(name, 255)
    return (r / 255.0, g / 255.0, b / 255.0, float(alpha))


def _render_target_pose_preview_3d(
    scene: SceneState,
    entries: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    out_dir: Path,
    robot_base_xyz: np.ndarray,
    *,
    draw_robot_reach: bool = True,
    robot_label: str = "robot base",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(11.8, 8.6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    robot_base_xyz = np.asarray(robot_base_xyz, dtype=np.float32).reshape(3)
    all_points: list[np.ndarray] = [np.vstack([robot_base_xyz, robot_base_xyz + [0.0, 0.0, 0.20]]).astype(np.float32)]

    current_entries = [entry for entry in entries if entry.get("kind") == "current"]
    target_entries = [entry for entry in entries if entry.get("kind") == "target"]
    for entry in current_entries:
        T = np.asarray(entry["T"], dtype=np.float32).reshape(4, 4)
        spec = str(entry["spec"])
        pts = _world_points(T, spec)
        all_points.append(pts)
        faces = _cuboid_faces(pts)
        alpha = 0.18 if spec == "desk" else 0.24
        coll = Poly3DCollection(
            faces,
            facecolors=_mpl_rgba(spec, alpha),
            edgecolors=_mpl_rgba(spec, 0.48),
            linewidths=0.8,
        )
        ax.add_collection3d(coll)
        p = T[:3, 3]
        ax.text(float(p[0]), float(p[1]), float(p[2] + 0.025), str(entry["object_id"]), fontsize=8, color="#334155")

    for entry in target_entries:
        T = np.asarray(entry["T"], dtype=np.float32).reshape(4, 4)
        spec = str(entry["spec"])
        pts = _world_points(T, spec)
        all_points.append(pts)
        faces = _cuboid_faces(pts)
        coll = Poly3DCollection(
            faces,
            facecolors=_mpl_rgba(spec, 0.34),
            edgecolors=(0.10, 0.33, 0.92, 1.0),
            linewidths=2.0,
        )
        ax.add_collection3d(coll)
        p = T[:3, 3]
        ax.scatter([p[0]], [p[1]], [p[2]], s=32, color="#2563eb", depthshade=False)
        for axis_idx, color in ((0, "#dc2626"), (1, "#16a34a"), (2, "#2563eb")):
            axis = T[:3, axis_idx].astype(np.float32)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-6:
                axis = axis / norm * 0.055
                ax.plot([p[0], p[0] + axis[0]], [p[1], p[1] + axis[1]], [p[2], p[2] + axis[2]], color=color, linewidth=2.6)
        ax.text(float(p[0]), float(p[1]), float(p[2] + 0.045), f"{entry['index']} {entry['object_id']}", fontsize=9, color="#1d4ed8")
        start = scene.get(str(entry["object_id"])).position
        ax.plot([start[0], p[0]], [start[1], p[1]], [start[2], p[2]], color="#2563eb", linewidth=1.8, alpha=0.62)

    # Simplified robot base at world origin. The offline LLM preview does not have current joint qpos.
    theta = np.linspace(0, 2 * np.pi, 36)
    base_r = 0.040
    z = np.linspace(0.0, 0.18, 2)
    tt, zz = np.meshgrid(theta, z)
    xx = robot_base_xyz[0] + base_r * np.cos(tt)
    yy = robot_base_xyz[1] + base_r * np.sin(tt)
    zz = robot_base_xyz[2] + zz
    ax.plot_surface(xx, yy, zz, color="#0f172a", alpha=0.42, linewidth=0, shade=True)
    ax.plot(
        [robot_base_xyz[0], robot_base_xyz[0] + 0.10],
        [robot_base_xyz[1], robot_base_xyz[1]],
        [robot_base_xyz[2] + 0.18, robot_base_xyz[2] + 0.18],
        color="#0f172a",
        linewidth=4,
    )
    ax.text(float(robot_base_xyz[0] + 0.015), float(robot_base_xyz[1] + 0.015), float(robot_base_xyz[2] + 0.22), robot_label, fontsize=9, color="#0f172a")
    if draw_robot_reach:
        for radius in (0.25, 0.45, 0.65):
            ax.plot(
                robot_base_xyz[0] + radius * np.cos(theta),
                robot_base_xyz[1] + radius * np.sin(theta),
                np.full_like(theta, robot_base_xyz[2]),
                color="#64748b",
                alpha=0.30,
                linestyle="--",
                linewidth=1.0,
            )
        all_points.append(
            np.asarray(
                [
                    robot_base_xyz + [0.65, 0.65, 0.0],
                    robot_base_xyz + [-0.65, -0.65, 0.0],
                ],
                dtype=np.float32,
            )
        )

    pts_all = np.concatenate(all_points, axis=0)
    mins = np.min(pts_all, axis=0)
    maxs = np.max(pts_all, axis=0)
    center = (mins + maxs) * 0.5
    span = np.max(np.maximum(maxs - mins, 0.18))
    span *= 1.18
    ax.set_xlim(center[0] - span * 0.5, center[0] + span * 0.5)
    ax.set_ylim(center[1] - span * 0.5, center[1] + span * 0.5)
    ax.set_zlim(0.0, max(0.36, float(maxs[2] + 0.08)))
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_zlabel("world Z (m)")
    ax.set_title("LLM target pose preview (3D)")
    ax.view_init(elev=28, azim=-54)
    ax.grid(True, alpha=0.24)
    try:
        ax.set_box_aspect((1, 1, 0.45))
    except Exception:
        pass
    fig.tight_layout()
    path = out_dir / "target_pose_preview_3d.png"
    fig.savefig(path, facecolor="#f8fafc", bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _render_target_pose_preview_3d_pil(
    scene: SceneState,
    entries: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    out_dir: Path,
    robot_base_xyz: np.ndarray,
    note: str = "",
    *,
    draw_robot_reach: bool = True,
    robot_label: str = "robot base",
) -> str:
    """Dependency-light isometric preview used when matplotlib is unavailable."""
    from PIL import Image, ImageDraw

    width, height = 1180, 860
    margin = 74
    robot_base_xyz = np.asarray(robot_base_xyz, dtype=np.float32).reshape(3)

    robot_points = [robot_base_xyz, robot_base_xyz + [0.0, 0.0, 0.10], robot_base_xyz + [0.12, 0.0, 0.10]]
    if draw_robot_reach:
        robot_points.extend([robot_base_xyz + [0.65, 0.65, 0.0], robot_base_xyz + [-0.65, -0.65, 0.0]])
    all_points: list[np.ndarray] = [np.vstack(robot_points).astype(np.float32)]
    for entry in entries:
        all_points.append(_world_points(np.asarray(entry["T"], dtype=np.float32).reshape(4, 4), str(entry["spec"])))
    pts_all = np.concatenate(all_points, axis=0)

    def project_raw(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float32).reshape(3)
        u = (float(p[0]) - float(p[1])) * 0.88
        v = (float(p[0]) + float(p[1])) * 0.42 - float(p[2]) * 1.85
        return np.asarray([u, v], dtype=np.float32)

    raw = np.asarray([project_raw(p) for p in pts_all], dtype=np.float32)
    min_uv = raw.min(axis=0)
    max_uv = raw.max(axis=0)
    span = np.maximum(max_uv - min_uv, 1e-3)
    scale = min((width - 2 * margin) / float(span[0]), (height - 2 * margin) / float(span[1]))
    offset = np.asarray(
        [
            margin - float(min_uv[0]) * scale + ((width - 2 * margin) - float(span[0]) * scale) * 0.5,
            margin - float(min_uv[1]) * scale + ((height - 2 * margin) - float(span[1]) * scale) * 0.5,
        ],
        dtype=np.float32,
    )

    def to_px(p: np.ndarray) -> tuple[float, float]:
        uv = project_raw(p) * scale + offset
        return float(uv[0]), float(uv[1])

    def depth(p: np.ndarray) -> float:
        p = np.asarray(p, dtype=np.float32).reshape(-1, 3)
        return float(np.mean(p[:, 0] + p[:, 1] + p[:, 2] * 0.25))

    img = Image.new("RGBA", (width, height), (248, 250, 252, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, 0, width, height), fill=(248, 250, 252, 255))
    draw.text((28, 24), "LLM target pose preview (3D fallback)", fill=(15, 23, 42, 255))
    if note:
        draw.text((28, 48), "fallback renderer active; see manifest for matplotlib details", fill=(100, 116, 139, 255))

    # Reach rings and simplified robot base.
    rb = robot_base_xyz
    if draw_robot_reach:
        for radius in (0.25, 0.45, 0.65):
            pts = [to_px(rb + [radius * math.cos(t), radius * math.sin(t), 0.0]) for t in np.linspace(0, 2 * math.pi, 96)]
            draw.line(pts + pts[:1], fill=(100, 116, 139, 90), width=1)
    bx, by = to_px(rb)
    bz = to_px(rb + [0.0, 0.0, 0.10])
    draw.ellipse((bx - 16, by - 10, bx + 16, by + 10), fill=(15, 23, 42, 180), outline=(255, 255, 255, 220), width=2)
    draw.line((bx, by, bz[0], bz[1]), fill=(15, 23, 42, 190), width=4)
    arm_tip = to_px(rb + [0.12, 0.0, 0.10])
    draw.line((bz[0], bz[1], arm_tip[0], arm_tip[1]), fill=(15, 23, 42, 220), width=5)
    _draw_label(draw, (bx + 12, by + 10), robot_label, fill=(255, 255, 255, 255), bg=(15, 23, 42, 220))

    face_items: list[tuple[float, bool, str, str, int, np.ndarray]] = []
    for entry in entries:
        T = np.asarray(entry["T"], dtype=np.float32).reshape(4, 4)
        spec = str(entry["spec"])
        pts = _world_points(T, spec)
        is_target = entry.get("kind") == "target"
        idx = int(entry.get("index") or 0)
        for face in _cuboid_faces(pts):
            face_items.append((depth(face), is_target, str(entry.get("object_id", "")), spec, idx, face))
    face_items.sort(key=lambda item: item[0])

    for _, is_target, object_id, spec, idx, face in face_items:
        poly = [to_px(p) for p in face]
        fill = _preview_color(spec, 92 if is_target else 54)
        outline = (37, 99, 235, 240) if is_target else _preview_color(spec, 140)
        draw.polygon(poly, fill=fill)
        draw.line(poly + poly[:1], fill=outline, width=3 if is_target else 1)

    for entry in entries:
        T = np.asarray(entry["T"], dtype=np.float32).reshape(4, 4)
        p = T[:3, 3]
        x, y = to_px(p)
        if entry.get("kind") == "target":
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(37, 99, 235, 255))
            for axis_idx, color in ((0, (220, 38, 38, 255)), (1, (22, 163, 74, 255)), (2, (37, 99, 235, 255))):
                axis = T[:3, axis_idx].astype(np.float32)
                norm = float(np.linalg.norm(axis))
                if norm > 1e-6:
                    end = p + axis / norm * 0.055
                    ex, ey = to_px(end)
                    draw.line((x, y, ex, ey), fill=color, width=3)
            start = scene.get(str(entry["object_id"])).position
            sx, sy = to_px(start)
            draw.line((sx, sy, x, y), fill=(37, 99, 235, 150), width=2)
            _draw_label(draw, (x + 10, y - 22), f"{entry['index']} {entry['object_id']}", fill=(255, 255, 255, 255), bg=(37, 99, 235, 230))
        else:
            _draw_label(draw, (x + 7, y + 4), str(entry["object_id"]), fill=(51, 65, 85, 255), bg=(255, 255, 255, 190))

    legend_x, legend_y = width - 330, 74
    draw.rounded_rectangle((legend_x, legend_y, width - 28, min(height - 28, legend_y + 54 + 44 * len(steps))), radius=10, fill=(255, 255, 255, 226), outline=(203, 213, 225, 255))
    draw.text((legend_x + 18, legend_y + 14), "Target steps", fill=(15, 23, 42, 255))
    for step in steps[:12]:
        idx = int(step["index"])
        y = legend_y + 34 + idx * 40
        spec = str(step.get("source_spec") or scene.get(str(step["source_id"])).spec_name)
        draw.ellipse((legend_x + 18, y + 6, legend_x + 32, y + 20), fill=_preview_color(spec, 220))
        xyz = step.get("target_pose_xyz_m") or []
        xyz_text = ", ".join(f"{float(v):.2f}" for v in xyz[:3])
        draw.text((legend_x + 42, y), f"{idx}. {step.get('source_id', '')}", fill=(15, 23, 42, 255))
        draw.text((legend_x + 42, y + 17), xyz_text, fill=(100, 116, 139, 255))

    path = out_dir / "target_pose_preview_3d.png"
    img.convert("RGB").save(path)
    return str(path)


def render_target_pose_preview(scene: SceneState, result: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Render a lightweight top-down preview of materialized LLM target poses."""
    from PIL import Image, ImageDraw

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps = list(result.get("steps") or [])
    if not steps:
        return {"ok": False, "reason": "no steps"}

    entries: list[dict[str, Any]] = []
    bounds_points: list[np.ndarray] = []
    robot_base_xyz, robot_base_source = _scene_robot_base_world_xyz_with_source(scene)
    robot_base_is_explicit = robot_base_source != "fallback"
    robot_label = "robot base" if robot_base_is_explicit else "robot base default"
    for obj in scene.objects.values():
        try:
            hull = _footprint_hull(obj.T_world_obj, obj.spec_name)
        except Exception:
            hull = np.asarray(
                [
                    obj.position[:2] + [-0.015, -0.015],
                    obj.position[:2] + [0.015, -0.015],
                    obj.position[:2] + [0.015, 0.015],
                    obj.position[:2] + [-0.015, 0.015],
                ],
                dtype=np.float32,
            )
        entries.append({"kind": "current", "object_id": obj.object_id, "spec": obj.spec_name, "T": obj.T_world_obj, "hull": hull})
        bounds_points.append(hull)
    robot_base_xy = robot_base_xyz[:2].astype(np.float32)
    bounds_points.append(robot_base_xy.reshape(1, 2))
    if robot_base_is_explicit:
        robot_extent = np.asarray(
            [
                robot_base_xy + [-0.68, -0.68],
                robot_base_xy + [0.68, 0.68],
                robot_base_xy,
            ],
            dtype=np.float32,
        )
        bounds_points.append(robot_extent)

    for step in steps:
        source_id = str(step["source_id"])
        source = scene.get(source_id)
        T_goal = np.asarray(step["target_pose"], dtype=np.float32).reshape(4, 4)
        try:
            hull = _footprint_hull(T_goal, source.spec_name)
        except Exception:
            p = T_goal[:2, 3]
            hull = np.asarray([p + [-0.015, -0.015], p + [0.015, -0.015], p + [0.015, 0.015], p + [-0.015, 0.015]], dtype=np.float32)
        entries.append(
            {
                "kind": "target",
                "index": int(step["index"]),
                "object_id": source_id,
                "spec": source.spec_name,
                "T": T_goal,
                "hull": hull,
                "description": step.get("description", ""),
            }
        )
        bounds_points.append(hull)

    all_xy = np.concatenate(bounds_points, axis=0)
    min_xy = np.min(all_xy, axis=0)
    max_xy = np.max(all_xy, axis=0)
    center = (min_xy + max_xy) * 0.5
    span = np.maximum(max_xy - min_xy, 0.20)
    margin = np.maximum(span * 0.18, 0.05)
    min_xy = center - span * 0.5 - margin
    max_xy = center + span * 0.5 + margin

    width, height = 1180, 860
    pad_left, pad_right, pad_top, pad_bottom = 70, 290, 55, 70
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    sx = plot_w / max(float(max_xy[0] - min_xy[0]), 1e-6)
    sy = plot_h / max(float(max_xy[1] - min_xy[1]), 1e-6)
    scale = min(sx, sy)
    used_w = scale * float(max_xy[0] - min_xy[0])
    used_h = scale * float(max_xy[1] - min_xy[1])
    ox = pad_left + (plot_w - used_w) * 0.5
    oy = pad_top + (plot_h - used_h) * 0.5

    def to_px(xy: np.ndarray | list[float] | tuple[float, float]) -> tuple[float, float]:
        arr = np.asarray(xy, dtype=np.float32).reshape(2)
        return (
            float(ox + (arr[0] - min_xy[0]) * scale),
            float(oy + (max_xy[1] - arr[1]) * scale),
        )

    def poly_px(hull: np.ndarray) -> list[tuple[float, float]]:
        return [to_px(pt) for pt in np.asarray(hull, dtype=np.float32).reshape(-1, 2)]

    img = Image.new("RGBA", (width, height), (248, 250, 252, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((pad_left, pad_top, width - pad_right, height - pad_bottom), outline=(190, 198, 209, 255), width=2)

    # Grid in world meters.
    grid_step = 0.05
    x0 = math.floor(float(min_xy[0]) / grid_step) * grid_step
    x1 = math.ceil(float(max_xy[0]) / grid_step) * grid_step
    y0 = math.floor(float(min_xy[1]) / grid_step) * grid_step
    y1 = math.ceil(float(max_xy[1]) / grid_step) * grid_step
    gx = x0
    while gx <= x1 + 1e-6:
        p0 = to_px([gx, min_xy[1]])
        p1 = to_px([gx, max_xy[1]])
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(226, 232, 240, 255), width=1)
        if abs((gx / 0.10) - round(gx / 0.10)) < 1e-4:
            draw.text((p0[0] - 18, height - pad_bottom + 10), f"{gx:.2f}", fill=(100, 116, 139, 255))
        gx += grid_step
    gy = y0
    while gy <= y1 + 1e-6:
        p0 = to_px([min_xy[0], gy])
        p1 = to_px([max_xy[0], gy])
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(226, 232, 240, 255), width=1)
        if abs((gy / 0.10) - round(gy / 0.10)) < 1e-4:
            draw.text((10, p0[1] - 6), f"{gy:.2f}", fill=(100, 116, 139, 255))
        gy += grid_step

    draw.text((pad_left, 20), "LLM target pose preview (top view, meters)", fill=(15, 23, 42, 255))
    draw.text((width - pad_right + 22, 20), "steps", fill=(15, 23, 42, 255))
    _draw_table_frame_axes_2d(draw, to_px, min_xy, max_xy)
    _draw_robot_base_2d(draw, to_px, robot_base_xyz, draw_reach=robot_base_is_explicit, label=robot_label)

    # Current scene first.
    for entry in entries:
        if entry["kind"] != "current":
            continue
        color = _preview_color(entry["spec"], 65 if entry["spec"] == "desk" else 92)
        outline = _preview_color(entry["spec"], 210)
        pts = poly_px(entry["hull"])
        if len(pts) >= 3:
            draw.polygon(pts, fill=color, outline=outline)
        center_px = to_px(np.asarray(entry["T"], dtype=np.float32)[:2, 3])
        _draw_label(draw, (center_px[0] + 4, center_px[1] + 4), str(entry["object_id"]), fill=(51, 65, 85, 255), bg=(255, 255, 255, 180))

    # Arrows from current pose to target, then target ghosts.
    for entry in entries:
        if entry["kind"] != "target":
            continue
        obj = scene.get(entry["object_id"])
        start_px = to_px(obj.position[:2])
        goal_xy = np.asarray(entry["T"], dtype=np.float32)[:2, 3]
        end_px = to_px(goal_xy)
        draw.line((start_px[0], start_px[1], end_px[0], end_px[1]), fill=(37, 99, 235, 190), width=3)
        dx, dy = end_px[0] - start_px[0], end_px[1] - start_px[1]
        norm = math.hypot(dx, dy)
        if norm > 1.0:
            ux, uy = dx / norm, dy / norm
            left = (end_px[0] - 12 * ux - 6 * uy, end_px[1] - 12 * uy + 6 * ux)
            right = (end_px[0] - 12 * ux + 6 * uy, end_px[1] - 12 * uy - 6 * ux)
            draw.polygon([end_px, left, right], fill=(37, 99, 235, 210))

        pts = poly_px(entry["hull"])
        fill = _preview_color(entry["spec"], 88)
        outline = (37, 99, 235, 255)
        if len(pts) >= 3:
            draw.polygon(pts, fill=fill)
            draw.line(pts + [pts[0]], fill=outline, width=4)

        T = np.asarray(entry["T"], dtype=np.float32).reshape(4, 4)
        center_px = to_px(T[:2, 3])
        for axis_idx, axis_color in ((0, (220, 38, 38, 255)), (1, (22, 163, 74, 255))):
            axis_xy = T[:2, axis_idx].astype(np.float32)
            axis_norm = float(np.linalg.norm(axis_xy))
            if axis_norm > 1e-6:
                axis_xy = axis_xy / axis_norm * 0.045
                tip = to_px(T[:2, 3] + axis_xy)
                draw.line((center_px[0], center_px[1], tip[0], tip[1]), fill=axis_color, width=4)
                draw.ellipse((tip[0] - 3, tip[1] - 3, tip[0] + 3, tip[1] + 3), fill=axis_color)
        _draw_label(draw, (center_px[0] + 8, center_px[1] - 18), f"{entry['index']} {entry['object_id']}", fill=(15, 23, 42, 255), bg=(219, 234, 254, 226))

    legend_y = 52
    for step in steps:
        idx = int(step["index"])
        y = legend_y + (idx - 1) * 58
        if y > height - 72:
            break
        spec = str(step.get("source_spec") or scene.get(str(step["source_id"])).spec_name)
        draw.rounded_rectangle((width - pad_right + 20, y, width - 24, y + 46), radius=7, fill=(255, 255, 255, 230), outline=(216, 222, 233, 255))
        draw.ellipse((width - pad_right + 31, y + 15, width - pad_right + 47, y + 31), fill=_preview_color(spec, 220))
        draw.text((width - pad_right + 56, y + 8), f"{idx}. {step.get('source_id', '')}", fill=(15, 23, 42, 255))
        xyz = step.get("target_pose_xyz_m") or []
        xyz_text = ", ".join(f"{float(v):.3f}" for v in xyz[:3])
        draw.text((width - pad_right + 56, y + 26), xyz_text, fill=(100, 116, 139, 255))

    combined_path = out / "target_pose_preview.png"
    img.convert("RGB").save(combined_path)
    preview_3d_path = None
    preview_3d_error = None
    try:
        preview_3d_path = _render_target_pose_preview_3d(
            scene,
            entries,
            steps,
            out,
            robot_base_xyz,
            draw_robot_reach=robot_base_is_explicit,
            robot_label=robot_label,
        )
    except Exception as exc:
        preview_3d_error = repr(exc)
        try:
            preview_3d_path = _render_target_pose_preview_3d_pil(
                scene,
                entries,
                steps,
                out,
                robot_base_xyz,
                note=preview_3d_error,
                draw_robot_reach=robot_base_is_explicit,
                robot_label=robot_label,
            )
        except Exception as fallback_exc:
            preview_3d_error = f"{preview_3d_error}; fallback={fallback_exc!r}"

    preview = {
        "ok": True,
        "target_pose_preview_image": str(combined_path),
        "bounds_xy_m": [[float(min_xy[0]), float(min_xy[1])], [float(max_xy[0]), float(max_xy[1])]],
        "robot_base_world_xyz_m": [float(v) for v in robot_base_xyz],
        "robot_base_source": robot_base_source,
        "frame_axes": {"front": "+X", "left": "+Y"},
        "step_count": len(steps),
    }
    if preview_3d_path is not None:
        preview["target_pose_preview_3d_image"] = str(preview_3d_path)
    if preview_3d_error is not None:
        preview["target_pose_preview_3d_error"] = preview_3d_error
    return preview


def _command_for_step(args, scene_file: Path, rule_file: Path, step: ResolvedStep, scene: SceneState) -> list[str]:
    source = scene.get(step.source_id)
    tracked = [
        obj.spec_name
        for obj in scene.objects.values()
        if obj.object_id != step.source_id
    ]
    # Keep stable ordering and avoid repeated spec names.
    seen = set()
    tracked_unique = []
    for name in tracked:
        if name in seen:
            continue
        seen.add(name)
        tracked_unique.append(name)
    cmd = [
        *_direct_command_prefix(args),
        "--curobo-rm75-robot-cfg",
        str(args.curobo_rm75_robot_cfg),
        "--trajectory-preview-sleep",
        str(args.trajectory_preview_sleep),
        "--cycle-object-names",
        source.spec_name,
        "--tracked-scene-object-names",
        *tracked_unique,
        "--skip-foundationpose",
        "--fixed-scene-pose-file",
        str(scene_file),
        "--fixed-scene-strict",
        "--dynamic-place-rule-json",
        str(rule_file),
        "--render-mode",
        str(args.render_mode),
        "--auto-execute",
        "--cycle-order-targets",
        "--repeat-count",
        "1",
        "--no-next-cycle-plan-prefetch",
        "--dry-run-motion-window-scale",
        str(args.dry_run_motion_window_scale),
        "--real-control-hz",
        str(args.real_control_hz),
        "--real-max-delta-per-step",
        str(args.real_max_delta_per_step),
    ]
    if step.place_mode != "auto":
        cmd += ["--place-mode", step.place_mode]
    if _needs_gluestick_slot_fallback(
        source_spec=source.spec_name,
        operator=step.operator,
        description=step.description,
    ):
        cmd.append("--fast-chain-allow-legacy-fallback")
    if args.execute_real:
        cmd.append("--execute-real")
    return cmd


def _direct_command_prefix(args) -> list[str]:
    direct = str(args.direct_script)
    path = Path(direct).expanduser()
    if direct.endswith(".py") or path.exists():
        return [str(args.python), str(path)]
    return [str(args.python), "-m", direct]


def _needs_gluestick_slot_fallback(*, source_spec: str, operator: str, description: str = "") -> bool:
    if str(source_spec) != "gluestick":
        return False
    operator = str(operator)
    if operator in {"desk_slot", "collection_slot"}:
        return True
    return bool(operator == "collection_right_half" and "small-desk" in str(description))


def _combined_command_for_steps(
    args,
    scene_file: Path,
    rule_file: Path,
    steps: list[dict[str, Any]],
    scene: SceneState,
) -> list[str] | None:
    if not steps:
        return None
    source_specs = [str(step["source_spec"]) for step in steps]
    if len(set(source_specs)) != len(source_specs):
        return None
    gluestick_fallback_needed = any(
        _needs_gluestick_slot_fallback(
            source_spec=str(step.get("source_spec", "")),
            operator=str(step.get("operator", "")),
            description=str(step.get("description", "")),
        )
        for step in steps
    )
    if gluestick_fallback_needed and any(
        str(step.get("operator", "")) not in {"desk_slot", "collection_slot", "collection_right_half"}
        for step in steps
    ):
        return None
    scene_specs = [obj.spec_name for obj in scene.objects.values()]
    if any(obj.object_id != obj.spec_name for obj in scene.objects.values()):
        return None
    collection_mode = any(
        str(step.get("operator", "")) in {"collection_slot", "collection_right_half"} for step in steps
    )

    tracked_unique: list[str] = []
    seen = set(source_specs)
    for spec_name in scene_specs:
        if spec_name in seen:
            continue
        seen.add(spec_name)
        tracked_unique.append(spec_name)

    place_modes = {str(step.get("place_mode", "surface_place")) for step in steps}
    cmd = [
        *_direct_command_prefix(args),
        "--curobo-rm75-robot-cfg",
        str(args.curobo_rm75_robot_cfg),
        "--trajectory-preview-sleep",
        str(args.trajectory_preview_sleep),
        "--cycle-object-names",
        *source_specs,
        "--tracked-scene-object-names",
        *tracked_unique,
        "--skip-foundationpose",
        "--fixed-scene-pose-file",
        str(scene_file),
        "--fixed-scene-strict",
        "--dynamic-place-rule-json",
        str(rule_file),
        "--render-mode",
        str(args.render_mode),
        "--auto-execute",
        "--repeat-count",
        str(len(source_specs)),
        "--no-next-cycle-plan-prefetch",
        "--dry-run-motion-window-scale",
        str(args.dry_run_motion_window_scale),
        "--real-control-hz",
        str(args.real_control_hz),
        "--real-max-delta-per-step",
        str(args.real_max_delta_per_step),
    ]
    if len(place_modes) == 1:
        only_mode = next(iter(place_modes))
        if only_mode != "auto":
            cmd += ["--place-mode", only_mode]
    if collection_mode:
        cmd.append("--cycle-order-targets")
    else:
        cmd.append("--cycle-order-targets")
        cmd.append("--allow-repeated-cycle-object-names")
    # Current four-IK fast-chain is intentionally strict. The gluestick small-desk
    # slot family is a known solvable case where fast-chain misses release IK, so
    # enable the slower legacy candidate fallback only for that narrow command.
    if gluestick_fallback_needed:
        cmd.append("--fast-chain-allow-legacy-fallback")
    if args.execute_real:
        cmd.append("--execute-real")
    return cmd


def materialize_plan(
    args,
    command: str,
    scene: SceneState,
    out_dir: Path,
    *,
    llm_plan_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_context_path = out_dir / "llm_scene_context_initial.json"
    llm_context_path.write_text(json.dumps(scene.context(), ensure_ascii=False, indent=2), encoding="utf-8")
    llm_call_info: dict[str, Any] | None = None
    raw_external_plan: dict[str, Any] | None = None
    if llm_plan_override is not None:
        llm_plan = llm_plan_override
        llm_plan_source = "provided_json"
    elif str(getattr(args, "llm_provider", "mock") or "mock").strip().lower() not in {"", "mock", "local_mock"}:
        raw_external_plan, llm_call_info = call_external_llm_plan(args, command, scene, out_dir)
        llm_plan = external_llm_plan_to_internal(raw_external_plan, scene)
        llm_plan_source = str(getattr(args, "llm_provider", "external"))
    else:
        llm_plan = mock_llm_parse(command, scene)
        llm_plan_source = "mock"
    resolved_steps = resolve_plan(scene, llm_plan)
    working_scene = scene.copy()
    artifacts = []
    for idx, step in enumerate(resolved_steps, start=1):
        # Re-resolve single-step target against the updated scene for exchange steps already containing fixed poses.
        step.index = idx
        context_before = working_scene.context()
        context_before_path = out_dir / f"step_{idx:02d}_context_before.json"
        context_before_path.write_text(json.dumps(context_before, ensure_ascii=False, indent=2), encoding="utf-8")
        warnings = list(step.warnings) + _rough_overlap_warnings(working_scene, step)
        rule = _dynamic_rule_for_step(working_scene, step)
        rule_file = out_dir / f"step_{idx:02d}_dynamic_place_rule.json"
        rule_file.write_text(json.dumps(rule, ensure_ascii=False, indent=2), encoding="utf-8")
        scene_file = out_dir / f"step_{idx:02d}_fixed_scene.json"
        scene_file, command_scene_warnings = working_scene.write_command_fixed_scene(
            scene_file,
            source_id=step.source_id,
            target_id=step.target_object_id,
        )
        warnings.extend(command_scene_warnings)
        cmd = _command_for_step(args, scene_file, rule_file, step, working_scene)
        cmd_file = out_dir / f"step_{idx:02d}_command.sh"
        cmd_file.write_text(
            "#!/usr/bin/env bash\n"
            "set -e\n"
            f"cd {shlex.quote(str(APP_ROOT))}\n"
            + shlex.join(cmd)
            + "\n",
            encoding="utf-8",
        )
        artifacts.append(
            {
                "index": idx,
                "description": step.description,
                "source_id": step.source_id,
                "source_spec": working_scene.get(step.source_id).spec_name,
                "operator": step.operator,
                "target_object_id": step.target_object_id,
                "place_mode": step.place_mode,
                "primitive": step.primitive,
                "target_pose": np.asarray(step.target_pose, dtype=np.float32).tolist(),
                "target_pose_xyz_m": [round(float(v), 5) for v in step.target_pose[:3, 3]],
                "target_pose_quat_wxyz": [float(v) for v in mat2quat(step.target_pose[:3, :3])],
                "warnings": warnings,
                "rule_file": str(rule_file),
                "scene_file": str(scene_file),
                "context_before_file": str(context_before_path),
                "command_file": str(cmd_file),
                "command": shlex.join(cmd),
            }
        )
        working_scene.update_pose(step.source_id, step.target_pose, placed=True)
        context_after_path = out_dir / f"step_{idx:02d}_context_after.json"
        context_after_path.write_text(json.dumps(working_scene.context(), ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts[-1]["context_after_file"] = str(context_after_path)
    final_scene_file = out_dir / "final_predicted_scene.json"
    working_scene.write_fixed_scene(final_scene_file)
    combined_info: dict[str, Any] = {"available": False, "reason": "no steps"}
    if artifacts:
        source_specs = [str(item["source_spec"]) for item in artifacts]
        if any(obj.object_id != obj.spec_name for obj in scene.objects.values()):
            combined_info = {
                "available": False,
                "reason": "multi-instance scene ids are not safe for one combined low-level command yet",
            }
        else:
            combined_scene_file = out_dir / "combined_fixed_scene.json"
            scene.write_fixed_scene(combined_scene_file)
            combined_rule_file = out_dir / "combined_dynamic_place_rules.json"
            combined_rules = [
                json.loads(Path(item["rule_file"]).read_text(encoding="utf-8"))
                for item in artifacts
            ]
            combined_rule_file.write_text(
                json.dumps({"rules": combined_rules}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            combined_cmd = _combined_command_for_steps(
                args,
                combined_scene_file,
                combined_rule_file,
                artifacts,
                scene,
            )
            if combined_cmd is None:
                combined_info = {"available": False, "reason": "combined command builder rejected this plan"}
            else:
                combined_cmd_file = out_dir / "combined_command.sh"
                combined_cmd_file.write_text(
                    "#!/usr/bin/env bash\n"
                    "set -e\n"
                    f"cd {shlex.quote(str(APP_ROOT))}\n"
                    + shlex.join(combined_cmd)
                    + "\n",
                    encoding="utf-8",
                )
                combined_info = {
                    "available": True,
                    "scene_file": str(combined_scene_file),
                    "rule_file": str(combined_rule_file),
                    "command_file": str(combined_cmd_file),
                    "command": shlex.join(combined_cmd),
                    "step_count": len(artifacts),
                }
    result = {
        "command": command,
        "source_scene_file": str(scene.scene_file),
        "output_dir": str(out_dir),
        "llm_context_file": str(llm_context_path),
        "mock_llm_plan": llm_plan,
        "llm_plan": llm_plan,
        "llm_plan_source": llm_plan_source,
        "raw_external_llm_plan": raw_external_plan,
        "llm_call": llm_call_info,
        "step_count": len(artifacts),
        "steps": artifacts,
        "combined_command": combined_info,
        "final_predicted_scene_file": str(final_scene_file),
    }
    try:
        result["target_pose_preview"] = render_target_pose_preview(scene, result, out_dir)
    except Exception as exc:
        result["target_pose_preview"] = {"ok": False, "error": repr(exc)}
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    return result


def _extract_execution_artifacts(log_file: Path) -> dict[str, Any]:
    try:
        text = log_file.read_text(encoding="utf-8", errors="replace")
    except Exception:
        text = ""

    def _unique(paths: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for path in paths:
            path = path.strip()
            if not path or path in seen:
                continue
            seen.add(path)
            out.append(path)
        return out

    final_matches = re.findall(r"final success\s*=\s*(True|False)", text)
    cycle_matches = re.findall(r"cycle\s+\d+\s+success\s*=\s*(True|False)", text)
    final_success = None if not final_matches else final_matches[-1] == "True"
    cycle_successes = [item == "True" for item in cycle_matches]
    return {
        "execution_log_file": str(log_file),
        "final_success": final_success,
        "cycle_successes": cycle_successes,
        "failure_render_images": _unique(
            re.findall(r"\[inspect\]\s+saved failure render image:\s*(\S+)", text)
        ),
        "place_candidate_render_images": _unique(
            re.findall(r"\[inspect\]\s+saved place-candidate render image:\s*(\S+)", text)
        ),
    }


def _run_command_with_log(command_file: Path, log_file: Path) -> subprocess.CompletedProcess:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(APP_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(f"$ bash {command_file}\n")
        handle.flush()
        return subprocess.run(["bash", str(command_file)], cwd=str(APP_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)


def run_materialized_commands(result: dict[str, Any], *, stop_on_failure: bool = True) -> list[dict[str, Any]]:
    run_results: list[dict[str, Any]] = []
    output_dir = Path(result["output_dir"])
    combined = result.get("combined_command")
    if isinstance(combined, dict) and bool(combined.get("available")):
        command_file = Path(str(combined["command_file"]))
        log_file = output_dir / "combined_execution.log"
        print(f"[llm_orchestrator] RUN combined task: bash {command_file} > {log_file}")
        t0 = time.perf_counter()
        proc = _run_command_with_log(command_file, log_file)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        artifacts = _extract_execution_artifacts(log_file)
        item = {
            "index": 1,
            "combined": True,
            "step_count": int(combined.get("step_count", len(result.get("steps", [])))),
            "command_file": str(command_file),
            "returncode": int(proc.returncode),
            "elapsed_ms": float(elapsed_ms),
            **artifacts,
        }
        item["ok"] = proc.returncode == 0 and item.get("final_success") is True
        run_results.append(item)
        print(
            f"[llm_orchestrator] DONE combined task: "
            f"ok={item['ok']} returncode={proc.returncode} elapsed_ms={elapsed_ms:.1f} "
            f"failure_images={len(item['failure_render_images'])}"
        )
        result["run_results"] = run_results
        result["run_ok"] = bool(item["ok"])
        manifest = output_dir / "manifest.json"
        manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
        return run_results

    for step in result.get("steps", []):
        command_file = Path(step["command_file"])
        log_file = output_dir / f"step_{int(step['index']):02d}_execution.log"
        print(f"[llm_orchestrator] RUN step {step['index']}: bash {command_file} > {log_file}")
        t0 = time.perf_counter()
        proc = _run_command_with_log(command_file, log_file)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        artifacts = _extract_execution_artifacts(log_file)
        item = {
            "index": int(step["index"]),
            "command_file": str(command_file),
            "returncode": int(proc.returncode),
            "elapsed_ms": float(elapsed_ms),
            **artifacts,
        }
        item["ok"] = proc.returncode == 0 and item.get("final_success") is not False
        run_results.append(item)
        print(
            f"[llm_orchestrator] DONE step {step['index']}: "
            f"ok={item['ok']} returncode={proc.returncode} elapsed_ms={elapsed_ms:.1f} "
            f"failure_images={len(item['failure_render_images'])}"
        )
        if proc.returncode != 0 and stop_on_failure:
            break
    result["run_results"] = run_results
    result["run_ok"] = bool(run_results) and len(run_results) == len(result.get("steps", [])) and all(
        bool(item.get("ok")) for item in run_results
    )
    manifest = Path(result["output_dir"]) / "manifest.json"
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    return run_results


def print_materialized_result(
    result: dict[str, Any],
    *,
    case_index: int,
    command: str,
    show_manual_hint: bool = True,
) -> None:
    print(f"[llm_orchestrator] OK case {case_index:02d}: {command}")
    for step in result["steps"]:
        print(
            f"  step {step['index']}: {step['description']} "
            f"mode={step['place_mode']} target={step['target_pose_xyz_m']}"
        )
        if step["warnings"]:
            print(f"    warnings: {step['warnings']}")
        print(f"    command_file: {step['command_file']}")
    first_command = result["steps"][0]["command_file"] if result.get("steps") else None
    combined = result.get("combined_command")
    if isinstance(combined, dict) and bool(combined.get("available")):
        print(f"  combined_command_file: {combined['command_file']}")
        if show_manual_hint:
            print(f"  not executed. run manually: bash {combined['command_file']}")
    elif show_manual_hint and first_command:
        if isinstance(combined, dict) and combined.get("reason"):
            print(f"  combined command unavailable: {combined['reason']}")
        print(f"  not executed. run manually: bash {first_command}")


def completed_prefix_step_count(result: dict[str, Any], *, ran_commands: bool) -> int:
    steps = result.get("steps", [])
    if not ran_commands:
        return len(steps)
    run_results = result.get("run_results", [])
    if run_results and bool(run_results[0].get("combined")):
        return len(steps) if bool(run_results[0].get("ok")) else 0
    count = 0
    for item in run_results:
        if not bool(item.get("ok")):
            break
        count += 1
    return min(count, len(steps))


def apply_predicted_steps_to_scene(scene: SceneState, result: dict[str, Any], *, step_count: int | None = None) -> int:
    steps = list(result.get("steps", []))
    if step_count is not None:
        steps = steps[: max(0, int(step_count))]
    for step in steps:
        scene.update_pose(
            str(step["source_id"]),
            np.asarray(step["target_pose"], dtype=np.float32).reshape(4, 4),
            placed=True,
        )
    return len(steps)


def _execution_ok_for_summary(args, result: dict[str, Any]) -> bool:
    if not bool(getattr(args, "run_generated_commands", False)):
        return True
    return bool(result.get("run_ok", False))


def _run_failure_record(command: str, result: dict[str, Any]) -> dict[str, Any]:
    run_results = list(result.get("run_results", []))
    return {
        "command": command,
        "error": "generated command execution failed",
        "manifest_file": str(Path(result["output_dir"]) / "manifest.json"),
        "execution_log_files": [
            str(item["execution_log_file"]) for item in run_results if item.get("execution_log_file")
        ],
        "failure_render_images": [
            str(path) for item in run_results for path in list(item.get("failure_render_images", []) or [])
        ],
        "place_candidate_render_images": [
            str(path) for item in run_results for path in list(item.get("place_candidate_render_images", []) or [])
        ],
        "run_results": run_results,
    }


SELF_TEST_COMMANDS = [
    "把网球放进笔筒",
    "把网球扔进笔筒",
    "把网球放到笔筒里面",
    "把绿木块放到胡萝卜和刷子中间",
    "把绿色方块放在胡萝卜与刷子的中间",
    "把薯片盒原地立起来",
    "让薯片盒在原地竖起来",
    "把胶棒原地旋转45度",
    "胶棒转45度",
    "交换绿木块和网球的位置",
    "绿木块和网球换位置",
    "交绿木块和网球的位置",
    "把笔放进笔筒",
    "把胡萝卜放到桌子上",
    "把胶棒放到网球左边",
    "把绿木块放到3号位置",
    "把薯片盒原地立起来，然后把网球放在薯片盒上面",
    "把笔平放在笔筒上面",
    "把胶棒放在绿木块上面",
]


ADVANCED_SELF_TEST_COMMANDS = [
    "把薯片盒原地立起来，然后把网球放到薯片盒上面",
    "把胡萝卜放到绿木块和笔筒中间，尖头朝向笔筒",
    "把胶棒放在绿木块右边 5 厘米，和桌边平行",
    "把网球扔进笔筒，然后把笔靠在笔筒右侧",
    "交换网球和绿木块的位置，但不要碰到胡萝卜",
    "把所有还没放好的东西按从左到右的顺序放到桌子 1 到 6 号位",
    "把胶棒叠到绿木块上面，如果支撑不稳就放到绿木块旁边",
    "把网球放到离胡萝卜最近的空位",
    "清理桌面，把所有物体移到桌子右半边，不要互相重叠",
    "把红色的长条物体放到黄色球的后面",
    "先把挡路的东西挪开，再把网球放进笔筒",
    "整理桌面，把所有物体移到小桌子右半边，不要互相重叠",
    "把胡萝卜放到网球后面 10 厘米",
    "把刷子放到胶棒左侧 6 厘米，和桌边平行",
    "把绿木块放到离网球最近的空位",
    "把网球放到绿木块上面，然后把绿木块放到2号位置",
    "把胶棒放到5号位置，然后把胡萝卜放到胶棒前面",
    "把红色的长条物体放到黄色球后面 12 厘米",
    "把薯片盒放在笔筒和网球中间，然后把薯片盒原地旋转45度",
    "把网球放到小桌子上最近的空位",
]


def _make_run_dir(root: Path, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")[:48] or "command"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return root / f"{stamp}_{safe}_pid{os.getpid()}_{time.time_ns() % 1_000_000_000:09d}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mock-LLM natural-language pick-place orchestrator for fixed-scene offline testing."
    )
    parser.add_argument("--command", type=str, default=None, help="Natural-language command to parse and materialize.")
    parser.add_argument("--self-test", action="store_true", help="Run a battery of mock natural-language commands.")
    parser.add_argument("--advanced-self-test", action="store_true", help="Run the advanced natural-language command coverage set.")
    parser.add_argument("--print-llm-interface", action="store_true", help="Print the external LLM JSON interface and exit.")
    parser.add_argument("--llm-plan-json-file", type=Path, default=None, help="Use an external LLM JSON plan instead of the mock parser.")
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=DEFAULT_LLM_PROVIDER,
        choices=["mock", "openai-compatible", "deepseek"],
        help="LLM provider for natural-language parsing. mock keeps the local rule parser.",
    )
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL, help="Model name. DeepSeek default is deepseek-v4-flash.")
    parser.add_argument("--llm-api-base", type=str, default=DEFAULT_LLM_API_BASE, help="OpenAI-compatible API base URL.")
    parser.add_argument("--llm-api-key-env", type=str, default=DEFAULT_LLM_API_KEY_ENV, help="Environment variable containing the API key.")
    parser.add_argument("--llm-timeout-s", type=float, default=60.0)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-repair-attempts", type=int, default=1, help="Retry external LLM once by asking it to repair invalid JSON.")
    parser.add_argument("--no-llm-json-mode", dest="llm_json_mode", action="store_false", default=True)
    parser.add_argument("--llm-thinking", action="store_true", help="DeepSeek only: enable thinking mode. Disabled by default for fast JSON planning.")
    parser.add_argument("--llm-reasoning-effort", type=str, default="high", choices=["high", "max"])
    parser.add_argument("--validate-llm-plan-only", action="store_true", help="Validate/materialize the external LLM JSON plan without running commands.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep a prompt open and accept natural-language commands one by one.",
    )
    parser.add_argument("--fixed-scene-pose-file", type=str, default=None, help="Fixed scene JSON used as the current simulation state.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--direct-script", default=DEFAULT_DIRECT_SCRIPT)
    parser.add_argument("--curobo-rm75-robot-cfg", type=Path, default=DEFAULT_CUROBO_CFG)
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--trajectory-preview-sleep", type=float, default=0.08)
    parser.add_argument("--dry-run-motion-window-scale", type=float, default=1.0)
    parser.add_argument("--real-control-hz", type=int, default=30)
    parser.add_argument("--real-max-delta-per-step", type=float, default=0.1)
    parser.add_argument("--execute-real", action="store_true", help="Include --execute-real in generated commands.")
    parser.add_argument(
        "--run-generated-commands",
        action="store_true",
        help="Run each generated low-level command immediately after materializing it.",
    )
    parser.add_argument(
        "--keep-running-after-command-failure",
        action="store_true",
        help="When --run-generated-commands is set, continue to later generated steps after a failed step.",
    )
    parser.add_argument("--print-context", action="store_true", help="Print the LLM-readable scene context and exit.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.print_llm_interface:
        print(_llm_pick_place_interface_text())
        return
    if not args.fixed_scene_pose_file:
        raise ValueError("--fixed-scene-pose-file is required unless --print-llm-interface is used")
    scene = SceneState.load(args.fixed_scene_pose_file)
    if args.print_context:
        print(json.dumps(scene.context(), ensure_ascii=False, indent=2))
        return
    commands = []
    external_plan = None
    if args.llm_plan_json_file is not None:
        raw_plan = json.loads(Path(args.llm_plan_json_file).expanduser().read_text(encoding="utf-8"))
        external_plan = external_llm_plan_to_internal(raw_plan, scene)
        commands.append(str(raw_plan.get("user_command", args.llm_plan_json_file)))
    if args.self_test:
        commands.extend(SELF_TEST_COMMANDS)
    if args.advanced_self_test:
        commands.extend(ADVANCED_SELF_TEST_COMMANDS)
    if args.command:
        commands.append(args.command)
    interactive = bool(args.interactive) or not commands
    run_root = _make_run_dir(Path(args.output_root).expanduser().resolve(), "llm_pick_place")
    run_root.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    if interactive:
        print(f"[llm_orchestrator] interactive mode. run_dir={run_root}")
        print("[llm_orchestrator] 输入自然语言命令；输入 q/quit/exit/退出 结束；输入 context 打印当前场景。")
        idx = 0
        while True:
            try:
                command = input("llm-pick-place> ").strip()
            except EOFError:
                print()
                break
            if not command:
                continue
            if command.lower() in {"q", "quit", "exit"} or command in {"退出", "结束"}:
                break
            if command.lower() in {"context", "ctx"} or command in {"上下文", "场景"}:
                print(json.dumps(scene.context(), ensure_ascii=False, indent=2))
                continue
            idx += 1
            command_dir = run_root / f"case_{idx:02d}"
            try:
                result = materialize_plan(args, command, scene.copy(), command_dir)
                results.append(result)
                print_materialized_result(
                    result,
                    case_index=idx,
                    command=command,
                    show_manual_hint=not bool(args.run_generated_commands),
                )
                if args.run_generated_commands:
                    run_materialized_commands(
                        result,
                        stop_on_failure=not bool(args.keep_running_after_command_failure),
                    )
                    if not bool(result.get("run_ok", False)):
                        failures.append(_run_failure_record(command, result))
                step_count = completed_prefix_step_count(result, ran_commands=bool(args.run_generated_commands))
                applied = apply_predicted_steps_to_scene(scene, result, step_count=step_count)
                state_file = run_root / f"scene_after_case_{idx:02d}.json"
                scene.write_fixed_scene(state_file)
                print(f"[llm_orchestrator] updated interactive scene with {applied} step(s): {state_file}")
            except Exception as exc:
                failures.append({"command": command, "error": repr(exc)})
                print(f"[llm_orchestrator] FAIL case {idx:02d}: {command}: {exc}")
        summary = {
            "ok": len(failures) == 0,
            "interactive": True,
            "run_dir": str(run_root),
            "fixed_scene_pose_file": str(Path(args.fixed_scene_pose_file).expanduser().resolve()),
            "case_count": idx,
            "materialized_count": len(results),
            "success_count": sum(1 for item in results if _execution_ok_for_summary(args, item)),
            "failure_count": len(failures),
            "run_success_count": sum(1 for item in results if bool(item.get("run_ok", False))),
            "run_failure_count": sum(1 for item in results if bool(args.run_generated_commands) and not bool(item.get("run_ok", False))),
            "failures": failures,
            "manifest_files": [str(Path(item["output_dir"]) / "manifest.json") for item in results],
            "final_interactive_scene_file": str(run_root / "final_interactive_scene.json"),
        }
        scene.write_fixed_scene(run_root / "final_interactive_scene.json")
        summary_path = run_root / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[llm_orchestrator] summary: {summary_path}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if failures:
            raise SystemExit(1)
        return
    if not commands:
        raise ValueError("Provide --command, --self-test, or --interactive")
    for idx, command in enumerate(commands, start=1):
        command_dir = run_root / f"case_{idx:02d}"
        try:
            result = materialize_plan(
                args,
                command,
                scene.copy(),
                command_dir,
                llm_plan_override=external_plan if idx == 1 and external_plan is not None else None,
            )
            results.append(result)
            print_materialized_result(
                result,
                case_index=idx,
                command=command,
                show_manual_hint=not bool(args.run_generated_commands),
            )
            if args.run_generated_commands and not bool(args.validate_llm_plan_only):
                run_materialized_commands(
                    result,
                    stop_on_failure=not bool(args.keep_running_after_command_failure),
                )
                if not bool(result.get("run_ok", False)):
                    failures.append(_run_failure_record(command, result))
        except Exception as exc:
            failures.append({"command": command, "error": repr(exc)})
            print(f"[llm_orchestrator] FAIL case {idx:02d}: {command}: {exc}")
    summary = {
        "ok": len(failures) == 0,
        "run_dir": str(run_root),
        "fixed_scene_pose_file": str(Path(args.fixed_scene_pose_file).expanduser().resolve()),
        "case_count": len(commands),
        "materialized_count": len(results),
        "success_count": sum(1 for item in results if _execution_ok_for_summary(args, item)),
        "failure_count": len(failures),
        "run_success_count": sum(1 for item in results if bool(item.get("run_ok", False))),
        "run_failure_count": sum(1 for item in results if bool(args.run_generated_commands) and not bool(item.get("run_ok", False))),
        "failures": failures,
        "manifest_files": [str(Path(item["output_dir"]) / "manifest.json") for item in results],
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[llm_orchestrator] summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
