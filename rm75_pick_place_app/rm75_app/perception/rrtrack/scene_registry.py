from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from rm75_app.assets.object_specs import normalize_object_name


def build_scene_registry(payload: dict, active_object: str, active_instance_index: int = 0) -> dict:
    """Turn a SAM6D full-scene response into stable task-level object identities."""
    raw_results = payload.get("results") if isinstance(payload.get("results"), list) else [payload]
    counts: defaultdict[str, int] = defaultdict(int)
    wanted = normalize_object_name(active_object)
    objects: list[dict] = []
    active_id = None
    for item in raw_results:
        name = normalize_object_name(item.get("object_name")) or str(item.get("object_name") or "unknown")
        instance_index = counts[name]
        counts[name] += 1
        instance_id = f"{name}:{instance_index}"
        is_active = name == wanted and instance_index == int(active_instance_index)
        run_dir = Path(str(item.get("run_dir", ""))).expanduser()
        entry = {
            "instance_id": instance_id,
            "object_name": name,
            "instance_index": instance_index,
            "state": "active_tracking" if is_active else ("available" if item.get("ok", True) else "initialization_failed"),
            "ok": bool(item.get("ok", True)),
            "score": float(item.get("score", 0.0)),
            "T_cam_obj": item.get("T_cam_obj"),
            "run_dir": str(run_dir),
            "mask_path": str(run_dir / "sam6d_binary_mask.png") if str(run_dir) else None,
        }
        objects.append(entry)
        if is_active:
            active_id = instance_id
    if active_id is None:
        raise ValueError(f"scene initialization has no {active_object!r} instance {active_instance_index}")
    return {
        "schema_version": 1,
        "coordinate_frame": "camera_opencv",
        "active_instance_id": active_id,
        "objects": objects,
    }


def resolve_active_result(payload: dict, object_name: str, instance_index: int = 0) -> dict:
    results = payload.get("results") if isinstance(payload.get("results"), list) else [payload]
    wanted = normalize_object_name(object_name)
    matches = [item for item in results if normalize_object_name(item.get("object_name")) == wanted and item.get("ok", True)]
    if int(instance_index) < 0 or int(instance_index) >= len(matches):
        raise ValueError(f"SAM6D result has no {object_name!r} instance {instance_index}; found {len(matches)}")
    return matches[int(instance_index)]
