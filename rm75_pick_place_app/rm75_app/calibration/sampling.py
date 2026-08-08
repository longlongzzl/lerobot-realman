from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import save_json


def vector_deg_to_rad(value: Any) -> np.ndarray:
    return np.deg2rad(np.asarray(value, dtype=np.float64).reshape(-1))


def parse_joint_limits_deg(value: Any, joint_count: int) -> np.ndarray:
    if value is None:
        return np.repeat([[-360.0, 360.0]], joint_count, axis=0).astype(np.float64)
    limits = np.asarray(value, dtype=np.float64)
    if limits.shape == (2,):
        limits = np.repeat(limits.reshape(1, 2), joint_count, axis=0)
    limits = limits.reshape(joint_count, 2)
    if np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("joint lower limit is greater than upper limit")
    return limits


def generate_joint_jitter_samples(
    seed_qpos_rad: np.ndarray,
    *,
    count: int,
    jitter_deg: float | list[float] | np.ndarray,
    joint_limits_deg: Any = None,
    random_seed: int = 7,
    include_seed: bool = True,
) -> list[np.ndarray]:
    seed = np.asarray(seed_qpos_rad, dtype=np.float64).reshape(-1)
    joint_count = int(seed.size)
    if joint_count == 0:
        raise ValueError("seed_qpos_rad is empty")
    count = int(count)
    if count <= 0:
        return []
    jitter = np.asarray(jitter_deg, dtype=np.float64)
    if jitter.size == 1:
        jitter = np.repeat(float(jitter.reshape(-1)[0]), joint_count)
    jitter = np.abs(jitter.reshape(joint_count))
    limits_rad = np.deg2rad(parse_joint_limits_deg(joint_limits_deg, joint_count))
    rng = np.random.default_rng(int(random_seed))
    samples: list[np.ndarray] = []
    if include_seed:
        samples.append(np.clip(seed, limits_rad[:, 0], limits_rad[:, 1]))
    while len(samples) < count:
        delta = np.deg2rad(rng.uniform(-jitter, jitter))
        candidate = np.clip(seed + delta, limits_rad[:, 0], limits_rad[:, 1])
        samples.append(candidate.astype(np.float64))
    return samples[:count]


def save_qpos_plan(path: str | Path, samples_rad: list[np.ndarray], *, joint_names: list[str] | None = None) -> None:
    rows = []
    for idx, qpos in enumerate(samples_rad):
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        rows.append(
            {
                "index": idx,
                "qpos_rad": qpos,
                "qpos_deg": np.rad2deg(qpos),
                "joint_names": joint_names,
            }
        )
    save_json(path, rows)
