from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np


class TrackerState(str, Enum):
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    CORRECTING = "correcting"
    LOST = "lost"
    RECOVERING = "recovering"


@dataclass(frozen=True)
class FrameObservation:
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    frame_index: int = 0
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        if np.asarray(self.rgb).ndim != 3 or np.asarray(self.rgb).shape[2] != 3:
            raise ValueError("rgb must have shape HxWx3")
        if np.asarray(self.depth_m).shape != np.asarray(self.rgb).shape[:2]:
            raise ValueError("depth_m must match the RGB image size")
        if np.asarray(self.K).shape != (3, 3):
            raise ValueError("K must have shape 3x3")


@dataclass(frozen=True)
class SegmentationPrediction:
    mask: np.ndarray
    foreground_probability: np.ndarray | None = None


@dataclass(frozen=True)
class PoseEstimate:
    T_cam_obj: np.ndarray
    score: float = 0.0
    source: str = "local_refiner"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Agreement:
    precision: float
    support: float
    entropy: float
    mask_area: int
    rendered_area: int
    intersection_area: int


@dataclass
class RRTrackOutput:
    frame_index: int
    state: TrackerState
    T_cam_obj: np.ndarray | None
    mask: np.ndarray
    rendered_mask: np.ndarray | None
    agreement: Agreement
    accepted: bool
    event: str
    memory_updates: tuple[str, ...] = ()
    recovery_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def jsonable(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "state": self.state.value,
            "T_cam_obj": None if self.T_cam_obj is None else np.asarray(self.T_cam_obj).tolist(),
            "agreement": {
                "precision": float(self.agreement.precision),
                "support": float(self.agreement.support),
                "entropy": float(self.agreement.entropy),
                "mask_area": int(self.agreement.mask_area),
                "rendered_area": int(self.agreement.rendered_area),
                "intersection_area": int(self.agreement.intersection_area),
            },
            "accepted": bool(self.accepted),
            "event": self.event,
            "memory_updates": list(self.memory_updates),
            "recovery_source": self.recovery_source,
            "metadata": self.metadata,
        }


class MaskTracker(Protocol):
    def initialize(self, rgb: np.ndarray, mask: np.ndarray) -> SegmentationPrediction: ...

    def predict(self, rgb: np.ndarray) -> SegmentationPrediction: ...

    def inject(self, rgb: np.ndarray, mask: np.ndarray, *, long_term: bool) -> None: ...

    def clear_non_permanent_memory(self) -> None: ...


class PoseRefiner(Protocol):
    def refine(
        self,
        frame: FrameObservation,
        mask: np.ndarray,
        initial_pose: np.ndarray,
    ) -> PoseEstimate: ...

    def global_register(self, frame: FrameObservation, mask: np.ndarray) -> PoseEstimate | None: ...


class MaskRenderer(Protocol):
    def render(self, T_cam_obj: np.ndarray, K: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray: ...


class DescriptorEncoder(Protocol):
    def encode(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray: ...
