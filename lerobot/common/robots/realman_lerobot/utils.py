from __future__ import annotations
from typing import Dict, Tuple


def ensure_safe_goal_position(
    goal_present: Dict[str, Tuple[float, float]],
    max_relative_target: float,
) -> Dict[str, float]:
    """
    Clip each joint target to be within +/- max_relative_target of present position.

    Args:
        goal_present: {joint_name: (goal, present)}
        max_relative_target: maximum delta allowed (same unit as goal/present)

    Returns:
        clipped goal dict {joint_name: goal_clipped}
    """
    out: Dict[str, float] = {}
    m = float(max_relative_target)
    for k, (g, p) in goal_present.items():
        g = float(g)
        p = float(p)
        if g > p + m:
            g = p + m
        elif g < p - m:
            g = p - m
        out[k] = g
    return out