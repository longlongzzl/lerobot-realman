from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PerturbationCase:
    name: str
    dx: float
    dy: float
    yaw: float

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dx": float(self.dx),
            "dy": float(self.dy),
            "yaw": float(self.yaw),
        }


@dataclass
class AttemptResult:
    case: dict[str, Any]
    out_dir: str
    returncode: int
    summary: str = ""
    video: str = ""
    success: bool = False
    failed_at: str = ""
    failure_class: str = ""
    retryable: bool = True
    attempt: int = 1
    completed_roles: list[str] = field(default_factory=list)
    edge_errors_m: dict[str, float] = field(default_factory=dict)
    edge_angles_deg: dict[str, float] = field(default_factory=dict)
    attach_distances_m: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "out_dir": self.out_dir,
            "returncode": int(self.returncode),
            "summary": self.summary,
            "video": self.video,
            "success": bool(self.success),
            "failed_at": self.failed_at,
            "failure_class": self.failure_class,
            "retryable": bool(self.retryable),
            "attempt": int(self.attempt),
            "completed_roles": self.completed_roles,
            "edge_errors_m": self.edge_errors_m,
            "edge_angles_deg": self.edge_angles_deg,
            "attach_distances_m": self.attach_distances_m,
        }


@dataclass(frozen=True)
class VideoManifestItem:
    case_name: str
    attempt: int
    success: bool
    failure_class: str
    failed_at: str
    video: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "attempt": int(self.attempt),
            "success": bool(self.success),
            "failure_class": self.failure_class,
            "failed_at": self.failed_at,
            "video": str(self.video),
        }
