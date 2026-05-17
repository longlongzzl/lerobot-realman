from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RoleProfile:
    role: str
    source_summary: str
    candidate_index: int
    candidate: dict[str, Any]
    release_option: dict[str, Any]
    release_strategy: dict[str, Any]
    selected_bundle_score: list[Any]
    edge_alignment_after_release: dict[str, Any]
    connection_point_error_after_release: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source_summary": self.source_summary,
            "candidate_index": int(self.candidate_index),
            "candidate": self.candidate,
            "release_option": self.release_option,
            "release_strategy": self.release_strategy,
            "selected_bundle_score": self.selected_bundle_score,
            "edge_alignment_after_release": self.edge_alignment_after_release,
            "connection_point_error_after_release": self.connection_point_error_after_release,
        }


def extract_success_profiles(summary_path: Path) -> list[RoleProfile]:
    if not summary_path.exists():
        return []
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not bool((data.get("final") or {}).get("success", False)):
        return []
    profiles: list[RoleProfile] = []
    for report in data.get("reports", []):
        if not isinstance(report, dict) or not report.get("success"):
            continue
        profiles.append(
            RoleProfile(
                role=str(report.get("role", "")),
                source_summary=str(summary_path),
                candidate_index=int(report.get("candidate_index", -1)),
                candidate=dict(report.get("candidate") or {}),
                release_option=dict(report.get("release_option") or {}),
                release_strategy=dict(report.get("release_strategy") or {}),
                selected_bundle_score=list(report.get("selected_bundle_score") or []),
                edge_alignment_after_release=dict(report.get("edge_alignment_after_release") or {}),
                connection_point_error_after_release=dict(report.get("connection_point_error_after_release") or {}),
            )
        )
    return profiles


def update_profile_cache(cache_path: Path, summary_paths: list[Path]) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {"profiles": []}
    if cache_path.exists():
        try:
            existing = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {"profiles": []}
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in existing.get("profiles", []):
        if not isinstance(item, dict):
            continue
        key = (str(item.get("role", "")), str(item.get("source_summary", "")), int(item.get("candidate_index", -1)))
        by_key[key] = item
    for summary_path in summary_paths:
        for profile in extract_success_profiles(summary_path):
            item = profile.to_json()
            key = (profile.role, profile.source_summary, profile.candidate_index)
            by_key[key] = item
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "profile_count": len(by_key),
        "profiles": list(by_key.values()),
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
