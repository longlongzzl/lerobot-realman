from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class EMAMADThresholdManager:
    """Robust running statistics used by RRTrack's adaptive gates."""

    def __init__(self, *, alpha: float = 0.05, window: int = 100, min_samples: int = 8) -> None:
        self.alpha = float(alpha)
        self.window = int(window)
        self.min_samples = int(min_samples)
        self._ema: dict[str, float] = {}
        self._values: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

    def observe(self, metric: str, value: float) -> None:
        value = float(value)
        if not np.isfinite(value):
            return
        previous = self._ema.get(metric, value)
        self._ema[metric] = (1.0 - self.alpha) * previous + self.alpha * value
        self._values[metric].append(value)

    def lower_bound(
        self,
        metric: str,
        fallback: float,
        *,
        mad_scale: float = 2.5,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        values = self._values.get(metric)
        if values is None or len(values) < self.min_samples:
            return float(fallback)
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median))) * 1.4826
        center = float(self._ema.get(metric, median))
        return float(np.clip(center - float(mad_scale) * mad, minimum, maximum))

    def upper_bound(
        self,
        metric: str,
        fallback: float,
        *,
        mad_scale: float = 2.5,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        values = self._values.get(metric)
        if values is None or len(values) < self.min_samples:
            return float(fallback)
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median))) * 1.4826
        center = float(self._ema.get(metric, median))
        return float(np.clip(center + float(mad_scale) * mad, minimum, maximum))

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        result = {}
        for name, values in self._values.items():
            array = np.asarray(values, dtype=np.float64)
            median = float(np.median(array)) if array.size else 0.0
            result[name] = {
                "ema": float(self._ema.get(name, median)),
                "median": median,
                "mad": float(np.median(np.abs(array - median))) if array.size else 0.0,
                "samples": int(array.size),
            }
        return result
