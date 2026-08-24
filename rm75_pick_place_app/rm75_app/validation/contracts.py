from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: GateStatus
    summary: str
    checks: tuple[Mapping[str, Any], ...] = ()
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status is GateStatus.PASSED

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status.value,
            "passed": self.passed,
            "summary": self.summary,
            "checks": [dict(item) for item in self.checks],
            "artifacts": dict(self.artifacts),
            "error": self.error,
        }
