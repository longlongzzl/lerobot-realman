from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "best_versions.json"


CORE_FILES = [
    "jimu_pick_cube_env.py",
    "magnetic_snap.py",
    "curobo_rm75_planner.py",
    "plan_multi_wall_path_200step.py",
    "plan_second_layer_triangle_wall_path.py",
    "standard_four_wall_retry_build.py",
    "run_v7_attach10mm_robustness_matrix.py",
    "run_v9_full_physical_build.py",
]


PLANNER_FILES = [
    "assembly_planner/__init__.py",
    "assembly_planner/assembly_spec.py",
    "assembly_planner/full_build.py",
    "assembly_planner/profile_cache.py",
    "assembly_planner/retry_policy.py",
    "assembly_planner/schemas.py",
    "assembly_planner/second_layer_matrix.py",
    "assembly_planner/trajectory_score.py",
    "assembly_planner/version_registry.py",
    "assembly_planner/README.md",
]


@dataclass(frozen=True)
class BestVersionConfig:
    name: str
    source_run: Path
    note: str = ""
    snapshot_root: Path = ROOT
    include_videos: bool = True
    extra_files: list[Path] = field(default_factory=list)


def _copy_file(src: Path, dst: Path) -> str | None:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _copy_tree_files(src_root: Path, dst_root: Path, files: list[str]) -> list[str]:
    copied: list[str] = []
    for rel in files:
        copied_path = _copy_file(src_root / rel, dst_root / rel)
        if copied_path:
            copied.append(copied_path)
    return copied


def _collect_run_artifacts(source_run: Path, dst_root: Path, include_videos: bool) -> list[str]:
    copied: list[str] = []
    if not source_run.exists():
        return copied
    patterns = [
        "full_build_report.json",
        "matrix_results.json",
        "profile_cache.json",
        "video_manifest.json",
        "video_manifest.md",
        "standard_build_report.json",
        "triangle_wall_summary.json",
        "triangle_wall_manifest.json",
        "command.json",
        "stdout.log",
        "stderr.log",
        "phase_log.txt",
    ]
    if include_videos:
        patterns.extend(["*.mp4"])
    for pattern in patterns:
        for src in source_run.rglob(pattern):
            if not src.is_file():
                continue
            rel = src.relative_to(source_run)
            copied_path = _copy_file(src, dst_root / "run_artifacts" / rel)
            if copied_path:
                copied.append(copied_path)
    return copied


def _load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"versions": []}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"versions": []}


def record_best_version(config: BestVersionConfig) -> dict[str, Any]:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in config.name).strip("_")
    if not safe_name:
        raise ValueError("version name cannot be empty")
    snapshot_dir = Path(config.snapshot_root) / safe_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    copied_code = _copy_tree_files(ROOT, snapshot_dir / "code", CORE_FILES + PLANNER_FILES)
    copied_artifacts = _collect_run_artifacts(Path(config.source_run), snapshot_dir, bool(config.include_videos))
    copied_extra: list[str] = []
    for extra in config.extra_files:
        src = Path(extra)
        if not src.exists() or not src.is_file():
            continue
        copied = _copy_file(src, snapshot_dir / "extra" / src.name)
        if copied:
            copied_extra.append(copied)
    payload = {
        "name": safe_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_run": str(config.source_run),
        "snapshot_dir": str(snapshot_dir),
        "note": config.note,
        "copied_code": copied_code,
        "copied_artifacts": copied_artifacts,
        "copied_extra": copied_extra,
    }
    (snapshot_dir / "VERSION.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = [
        f"# {safe_name}",
        "",
        f"Created: {payload['created_at']}",
        f"Source run: `{config.source_run}`",
        "",
        "## Note",
        "",
        config.note or "No note provided.",
        "",
        "## Physical Boundary",
        "",
        "- Hard magnetic attachment distance must remain `0.010m`.",
        "- This snapshot records code and artifacts only; it does not delete or mutate previous runs.",
    ]
    (snapshot_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    registry = _load_registry()
    versions = [item for item in registry.get("versions", []) if item.get("name") != safe_name]
    versions.append(payload)
    registry = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "versions": versions,
    }
    REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
