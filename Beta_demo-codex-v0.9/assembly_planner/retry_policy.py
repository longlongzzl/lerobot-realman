from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureDecision:
    failure_class: str
    retryable: bool
    local_retry_first: bool
    restore_checkpoint: bool
    reason: str


def classify_failure(failed_at: str, returncode: int) -> FailureDecision:
    text = str(failed_at or "").lower()
    if returncode != 0 and not text:
        return FailureDecision("runtime_error", True, False, True, "process returned non-zero without summary failure text")
    if "missing_summary" in text or "summary_read_error" in text:
        return FailureDecision("runtime_error", True, False, True, "summary was not produced or could not be read")
    if "parent_edge_height" in text:
        return FailureDecision("scene_state_or_checkpoint", True, False, True, "target parent edge is not in a valid build pose")
    if "start_state" in text:
        return FailureDecision("scene_state_or_checkpoint", True, False, True, "robot start state is invalid for the planner world")
    if "ik" in text or "plan" in text or "preplace" in text:
        return FailureDecision("curobo_planning", True, True, False, "CuRobo could not find a usable segment")
    if "collision" in text or "retreat" in text:
        return FailureDecision("collision_or_retreat", True, True, True, "trajectory or retreat disturbed the scene")
    if "magnetic_capture" in text or "active_connection" in text:
        return FailureDecision("magnetic_capture", True, True, False, "panel reached the target area but did not form a valid magnetic connection")
    if "stability" in text or "final_failed" in text:
        return FailureDecision("post_release_stability", False, False, True, "post-release physics did not remain stable")
    if not text:
        return FailureDecision("", False, False, False, "success or no failure")
    return FailureDecision("unknown", True, True, True, "unrecognized failure")
