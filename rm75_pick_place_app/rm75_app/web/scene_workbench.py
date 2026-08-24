from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


KNOWNNESS_VALUES = {"known", "uncertain", "unknown", "ignored"}
ACTION_TYPES = {"process_unknown", "pick_place", "inspect"}


def _bbox_iou(a: Any, b: Any) -> float:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = (float(value) for value in a[:4])
        bx1, by1, bx2, by2 = (float(value) for value in b[:4])
    except (TypeError, ValueError):
        return 0.0
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 1e-6 else 0.0


def _item_bbox(item: dict[str, Any]) -> list[float] | None:
    value = item.get("mask_bbox")
    if value is None and isinstance(item.get("selected"), dict):
        value = item["selected"].get("box")
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [round(float(number), 2) for number in value[:4]]
    except (TypeError, ValueError):
        return None


def _score(item: dict[str, Any]) -> float | None:
    selected = item.get("selected") if isinstance(item.get("selected"), dict) else {}
    value = selected.get("model_score", item.get("score"))
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _copy_if_present(source: Any, destination: Path) -> str | None:
    if not source:
        return None
    path = Path(str(source)).expanduser()
    if not path.exists() or not path.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return str(destination.resolve())


class SceneWorkbench:
    """Persistent, thread-safe scene inventory and action-plan boundary for the web UI."""

    def __init__(
        self,
        root: str | Path,
        *,
        asset_names: Iterable[str],
        asset_dir: str | Path,
        bank_root: str | Path,
        python_executable: str = "python",
        initial_camera_transform: str | Path | None = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.state_path = self.root / "state.json"
        self.asset_names = sorted({str(name) for name in asset_names})
        self.asset_dir = Path(asset_dir).expanduser().resolve()
        self.bank_root = Path(bank_root).expanduser().resolve()
        self.python_executable = str(python_executable)
        self.initial_camera_transform = None if initial_camera_transform is None else Path(initial_camera_transform).expanduser().resolve()
        self._lock = threading.RLock()
        self._state = self._load()

    def _empty_state(self) -> dict[str, Any]:
        return {"schema_version": 1, "snapshot": None, "jobs": [], "plan": None}

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else self._empty_state()
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return self._empty_state()

    def _save(self) -> None:
        _atomic_json(self.state_path, self._state)

    def status(self, *, include_details: bool = False) -> dict[str, Any]:
        with self._lock:
            state = json.loads(json.dumps(self._state))
        if include_details:
            return state
        snapshot = state.get("snapshot") or {}
        for instance in snapshot.get("instances") or []:
            pose = instance.get("pose")
            if not isinstance(pose, dict):
                continue
            instance["pose"] = {
                key: pose.get(key)
                for key in ("ok", "score", "translation_m", "refine_applied")
                if pose.get(key) is not None
            }
        return state

    def _asset_readiness(self, name: str) -> dict[str, Any]:
        bank_candidates = sorted(self.bank_root.glob(f"{name}_*.npz"))
        return {
            "registered": name in self.asset_names,
            "rrtrack_bank_ready": bool(bank_candidates),
            "rrtrack_bank_path": str(bank_candidates[0]) if bank_candidates else None,
        }

    @staticmethod
    def _known_masks(result: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in result.get("known_mask_results") or []:
            if not isinstance(item, dict) or item.get("ok") is False:
                continue
            name = str(item.get("object_name") or "")
            if not name:
                raw_id = str(item.get("id") or "")
                name = raw_id.split("_p", 1)[0] if "_p" in raw_id else raw_id
            if not name:
                continue
            grouped.setdefault(name, []).append(item)
        output: list[tuple[str, int, dict[str, Any]]] = []
        for name, items in grouped.items():
            kept: list[dict[str, Any]] = []
            for item in sorted(items, key=lambda value: _score(value) or -1.0, reverse=True):
                if any(_bbox_iou(_item_bbox(item), _item_bbox(old)) >= 0.72 for old in kept):
                    continue
                kept.append(item)
            output.extend((name, index, item) for index, item in enumerate(kept))
        return output

    @staticmethod
    def _deduplicate_discovery(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted(
            (item for item in items if isinstance(item, dict) and item.get("ok") is not False and _item_bbox(item)),
            key=lambda item: (_score(item) or -1.0, int(item.get("mask_pixels") or 0)),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        for item in ranked:
            pixels = int(item.get("mask_pixels") or 0)
            bbox = _item_bbox(item)
            # The fixed scene camera is 640x480 today. Very large masks are
            # normally the table/background returned by the generic prompt.
            if pixels < 300 or pixels > 140_000 or bbox is None:
                continue
            if any(_bbox_iou(bbox, _item_bbox(old)) >= 0.72 for old in selected):
                continue
            selected.append(item)
        return selected

    @staticmethod
    def _reuse_instance_id(previous: list[dict[str, Any]], instance: dict[str, Any], used: set[str]) -> str | None:
        candidates = []
        for old in previous:
            old_id = str(old.get("instance_id") or "")
            if not old_id or old_id in used:
                continue
            same_asset = old.get("asset_name") == instance.get("asset_name")
            compatible = same_asset if instance.get("asset_name") else old.get("knownness") in {"unknown", "uncertain"}
            if compatible:
                candidates.append((_bbox_iou(old.get("bbox_xyxy"), instance.get("bbox_xyxy")), old_id))
        if not candidates:
            return None
        overlap, old_id = max(candidates)
        return old_id if overlap >= 0.35 else None

    def refresh(self, result: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            old_snapshot = self._state.get("snapshot") or {}
            previous = list(old_snapshot.get("instances") or [])
            version = int(old_snapshot.get("version") or 0) + 1
            pose_found = {str(name) for name in result.get("pose_found_objects") or []}
            known_masks = self._known_masks(result)
            pose_results = {
                (str(item.get("object_name") or ""), int(item.get("sam3_instance_index") or 0)): item
                for item in result.get("pose_results") or []
                if isinstance(item, dict)
            }
            instances: list[dict[str, Any]] = []

            for name, instance_index, item in known_masks:
                pose_detail = pose_results.get((name, instance_index))
                if pose_detail is None and instance_index == 0:
                    pose_detail = (result.get("pose_details") or {}).get(name) or {}
                pose_detail = dict(pose_detail or {})
                pose_ok = bool(pose_detail.get("ok", name in pose_found and instance_index == 0))
                readiness = self._asset_readiness(name)
                instances.append(
                    {
                        "asset_name": name,
                        "display_name": name,
                        "knownness": "known" if pose_ok and readiness["registered"] else "uncertain",
                        "confidence": pose_detail.get("score", _score(item)),
                        "reason": "资产匹配且6D位姿可用" if pose_ok else "已分割但6D位姿未通过",
                        "bbox_xyxy": _item_bbox(item),
                        "mask_path": item.get("mask_path"),
                        "pose": pose_detail,
                        "asset_instance_index": instance_index,
                        "asset": readiness,
                        "source": "known_asset_scan",
                    }
                )

            known_boxes = [item.get("bbox_xyxy") for item in instances]
            discovery = self._deduplicate_discovery(result.get("discovery_results") or [])
            for item in discovery:
                bbox = _item_bbox(item)
                if any(_bbox_iou(bbox, known_bbox) >= 0.45 for known_bbox in known_boxes):
                    continue
                instances.append(
                    {
                        "asset_name": None,
                        "display_name": str((item.get("vlm") or {}).get("noun_phrase") or item.get("prompt") or "未见物体"),
                        "knownness": "unknown",
                        "confidence": _score(item),
                        "reason": "桌面实例候选未与任何已知资产掩码匹配",
                        "bbox_xyxy": bbox,
                        "mask_path": item.get("mask_path"),
                        "pose": None,
                        "asset": {"registered": False, "rrtrack_bank_ready": False, "rrtrack_bank_path": None},
                        "source": "qwen_vl_sam3" if item.get("vlm") else "open_world_scan",
                        "vlm": item.get("vlm"),
                    }
                )

            used: set[str] = set()
            unknown_index = 0
            for instance in instances:
                reused = self._reuse_instance_id(previous, instance, used)
                if reused:
                    instance["instance_id"] = reused
                elif instance.get("asset_name"):
                    base = str(instance["asset_name"])
                    suffix = 1
                    candidate = f"{base}:{suffix}"
                    while candidate in used:
                        suffix += 1
                        candidate = f"{base}:{suffix}"
                    instance["instance_id"] = candidate
                else:
                    unknown_index += 1
                    instance["instance_id"] = f"unknown_{version:03d}_{unknown_index:02d}"
                used.add(instance["instance_id"])

            counts = {value: sum(item["knownness"] == value for item in instances) for value in KNOWNNESS_VALUES}
            snapshot = {
                "snapshot_id": f"scene_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
                "version": version,
                "created_at": time.time(),
                "frozen": False,
                "source_result_path": result.get("result_path"),
                "scene_dir": result.get("scene_dir"),
                "rgb_path": result.get("rgb_path"),
                "instances": instances,
                "counts": counts,
            }
            self._state["snapshot"] = snapshot
            self._state["plan"] = None
            self._save()
            return self.status()

    def update_instance(self, instance_id: str, *, knownness: str, asset_name: str | None = None) -> dict[str, Any]:
        if knownness not in KNOWNNESS_VALUES:
            raise ValueError(f"invalid knownness: {knownness}")
        if asset_name is not None and asset_name not in self.asset_names:
            raise ValueError(f"unknown asset: {asset_name}")
        with self._lock:
            snapshot = self._state.get("snapshot")
            if not snapshot or snapshot.get("frozen"):
                raise ValueError("scene snapshot is missing or frozen")
            instance = next((item for item in snapshot.get("instances") or [] if item.get("instance_id") == instance_id), None)
            if instance is None:
                raise KeyError(instance_id)
            instance["knownness"] = knownness
            instance["asset_name"] = asset_name if knownness in {"known", "uncertain"} else None
            readiness = self._asset_readiness(asset_name) if asset_name else {"registered": False, "rrtrack_bank_ready": False, "rrtrack_bank_path": None}
            pose_ok = bool((instance.get("pose") or {}).get("ok"))
            if knownness == "known" and asset_name and not pose_ok:
                instance["knownness"] = "uncertain"
                instance["reason"] = "人工指定资产候选，需重新扫描并通过6D位姿验证"
            else:
                instance["reason"] = "人工确认" if asset_name else "人工分类"
            instance["display_name"] = asset_name or ("忽略" if knownness == "ignored" else "未见物体")
            instance["asset"] = readiness
            snapshot["counts"] = {value: sum(item["knownness"] == value for item in snapshot["instances"]) for value in KNOWNNESS_VALUES}
            self._save()
            return self.status()

    def create_jobs(self, instance_ids: Iterable[str], *, provider: str = "observed") -> dict[str, Any]:
        if provider not in {"observed", "rayst3r"}:
            raise ValueError("provider must be observed or rayst3r")
        requested = list(dict.fromkeys(str(value) for value in instance_ids))
        with self._lock:
            snapshot = self._state.get("snapshot")
            if not snapshot:
                raise ValueError("scan the scene first")
            by_id = {item["instance_id"]: item for item in snapshot.get("instances") or []}
            missing = [value for value in requested if value not in by_id]
            if missing:
                raise KeyError(f"unknown instances: {missing}")
            invalid = [value for value in requested if by_id[value].get("knownness") != "unknown"]
            if invalid:
                raise ValueError(f"only unknown instances can be processed: {invalid}")
            jobs = []
            for instance_id in requested:
                instance = by_id[instance_id]
                job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                frame_dir = self.root / "jobs" / job_id / "initial_frame"
                scene_dir = Path(str(snapshot.get("scene_dir") or "")).expanduser()
                shared = scene_dir / "shared_frame"
                artifacts = {
                    "rgb_path": _copy_if_present(snapshot.get("rgb_path") or shared / "rgb.png", frame_dir / "rgb.png"),
                    "depth_path": _copy_if_present(shared / "depth.png", frame_dir / "depth.png"),
                    "camera_path": _copy_if_present(shared / "camera.json", frame_dir / "camera.json"),
                    "mask_path": _copy_if_present(instance.get("mask_path"), frame_dir / "mask.png"),
                }
                ready = all(artifacts.values())
                command = [
                    self.python_executable, "-m", "rm75_app", "openworld-geometry", "--",
                    "--instance-id", instance_id,
                    "--initial-frame-dir", str(frame_dir.resolve()),
                    "--provider", provider,
                    "--output-root", str((self.root / "jobs" / job_id / "geometry").resolve()),
                ]
                if self.initial_camera_transform is not None and self.initial_camera_transform.exists():
                    command.extend(["--initial-T-base-camera", str(self.initial_camera_transform)])
                job = {
                    "job_id": job_id,
                    "instance_id": instance_id,
                    "snapshot_id": snapshot["snapshot_id"],
                    "provider": provider,
                    "status": "capture_ready" if ready else "blocked",
                    "reason": "RGB-D/mask任务包已生成，等待几何重建" if ready else "初始帧缺少RGB、深度、相机参数或mask",
                    "created_at": time.time(),
                    "frame_dir": str(frame_dir.resolve()),
                    "artifacts": artifacts,
                    "command": command,
                }
                self._state.setdefault("jobs", []).append(job)
                jobs.append(job)
            self._state["jobs"] = self._state.get("jobs", [])[-100:]
            self._save()
            return {"jobs": jobs, "state": self.status()}

    def job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = next((item for item in self._state.get("jobs") or [] if item.get("job_id") == job_id), None)
            if job is None:
                raise KeyError(job_id)
            return json.loads(json.dumps(job))

    def update_job(self, job_id: str, *, status: str, reason: str, geometry_manifest: str | None = None) -> dict[str, Any]:
        with self._lock:
            job = next((item for item in self._state.get("jobs") or [] if item.get("job_id") == job_id), None)
            if job is None:
                raise KeyError(job_id)
            job["status"] = str(status)
            job["reason"] = str(reason)
            job["updated_at"] = time.time()
            if geometry_manifest is not None:
                job["geometry_manifest"] = str(geometry_manifest)
            self._save()
            return self.status()

    def save_plan(self, actions: Iterable[dict[str, Any]], *, freeze: bool = True) -> dict[str, Any]:
        with self._lock:
            snapshot = self._state.get("snapshot")
            if not snapshot:
                raise ValueError("scan the scene first")
            by_id = {item["instance_id"]: item for item in snapshot.get("instances") or []}
            normalized = []
            errors: list[str] = []
            planned_assets: set[str] = set()
            ready_geometry = {
                str(job.get("instance_id"))
                for job in self._state.get("jobs") or []
                if job.get("status") == "geometry_ready" and job.get("snapshot_id") == snapshot.get("snapshot_id")
            }
            for index, raw in enumerate(actions, start=1):
                action = dict(raw)
                action_type = str(action.get("type") or "")
                instance_id = str(action.get("instance_id") or "")
                if action_type not in ACTION_TYPES:
                    errors.append(f"step {index}: unsupported action {action_type!r}")
                    continue
                if instance_id not in by_id:
                    errors.append(f"step {index}: instance {instance_id!r} is not in this scene")
                    continue
                instance = by_id[instance_id]
                if action_type == "process_unknown":
                    if instance.get("knownness") != "unknown":
                        errors.append(f"step {index}: {instance_id} is not unknown")
                elif action_type == "pick_place":
                    if instance.get("knownness") == "unknown" and instance_id not in ready_geometry:
                        errors.append(f"step {index}: {instance_id} geometry is not ready")
                    if instance.get("knownness") in {"uncertain", "ignored"}:
                        errors.append(f"step {index}: {instance_id} is {instance.get('knownness')}")
                    if not action.get("destination"):
                        errors.append(f"step {index}: destination is required")
                    asset_name = str(instance.get("asset_name") or "")
                    if asset_name and asset_name in planned_assets:
                        errors.append(f"step {index}: duplicate asset {asset_name} is not yet addressable by the current pick-place executor")
                    if asset_name:
                        planned_assets.add(asset_name)
                normalized.append({
                    "step_id": f"step_{index:02d}",
                    "type": action_type,
                    "instance_id": instance_id,
                    "asset_name": instance.get("asset_name"),
                    "destination": action.get("destination"),
                })
            if not normalized:
                errors.append("plan has no actions")
            plan = {
                "plan_id": f"plan_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
                "snapshot_id": snapshot["snapshot_id"],
                "scene_version": snapshot["version"],
                "created_at": time.time(),
                "valid": not errors,
                "errors": errors,
                "actions": normalized,
            }
            self._state["plan"] = plan
            if freeze and plan["valid"]:
                snapshot["frozen"] = True
                snapshot["frozen_at"] = time.time()
            self._save()
            return self.status()

    def unfreeze(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self._state.get("snapshot")
            if snapshot:
                snapshot["frozen"] = False
                snapshot.pop("frozen_at", None)
            self._save()
            return self.status()
