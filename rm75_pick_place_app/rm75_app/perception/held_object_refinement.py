"""Optional wrist-camera refinement of a grasped object's TCP-relative 6D pose."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import numpy as np

from rm75_app.perception.rrtrack.models import FrameObservation, PoseEstimate
from rm75_app.planning.contracts import JointConfiguration


def _transform(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} must be homogeneous")
    return matrix.copy()


def _rotation_delta_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


@dataclass(frozen=True)
class HeldObjectRefinementConfig:
    min_mask_pixels: int = 80
    max_translation_delta_m: float = 0.035
    max_rotation_delta_deg: float = 18.0
    min_backend_score: float | None = None

    def __post_init__(self) -> None:
        if self.min_mask_pixels <= 0:
            raise ValueError("min_mask_pixels must be positive")
        if self.max_translation_delta_m <= 0.0 or self.max_rotation_delta_deg <= 0.0:
            raise ValueError("pose-delta gates must be positive")


@dataclass(frozen=True)
class HeldObjectRefinementRequest:
    object_name: str
    frame: FrameObservation
    mask: np.ndarray
    T_tcp_camera: np.ndarray
    prior_T_tcp_object: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.shape != np.asarray(self.frame.depth_m).shape:
            raise ValueError("held-object mask must match the frame depth shape")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "T_tcp_camera", _transform(self.T_tcp_camera, "T_tcp_camera"))
        object.__setattr__(
            self,
            "prior_T_tcp_object",
            _transform(self.prior_T_tcp_object, "prior_T_tcp_object"),
        )


@dataclass(frozen=True)
class HeldObjectRefinementUpdate:
    accepted: bool
    object_name: str
    T_tcp_object: np.ndarray | None
    translation_delta_m: float
    rotation_delta_deg: float
    reason: str
    source: str = "unknown"
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LocalPoseBackend(Protocol):
    def refine(
        self, frame: FrameObservation, mask: np.ndarray, initial_pose: np.ndarray
    ) -> PoseEstimate: ...


class HeldObjectPoseRefiner:
    """Convert a camera-frame local refinement into a gated TCP-relative pose."""

    def __init__(
        self,
        backend: LocalPoseBackend,
        config: HeldObjectRefinementConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or HeldObjectRefinementConfig()

    def refine(self, request: HeldObjectRefinementRequest) -> HeldObjectRefinementUpdate:
        pixels = int(np.count_nonzero(request.mask))
        if pixels < self.config.min_mask_pixels:
            return HeldObjectRefinementUpdate(
                False,
                request.object_name,
                None,
                0.0,
                0.0,
                f"mask_too_small:{pixels}",
            )
        # Both transforms are TCP-relative, so this prediction does not need
        # robot FK or a base-frame pose and remains valid while the arm moves.
        initial_T_camera_object = (
            np.linalg.inv(request.T_tcp_camera) @ request.prior_T_tcp_object
        )
        estimate = self.backend.refine(
            request.frame, request.mask, initial_T_camera_object
        )
        refined_T_camera_object = _transform(estimate.T_cam_obj, "T_camera_object")
        candidate = request.T_tcp_camera @ refined_T_camera_object
        translation_delta = float(
            np.linalg.norm(
                candidate[:3, 3] - request.prior_T_tcp_object[:3, 3]
            )
        )
        rotation_delta = _rotation_delta_deg(request.prior_T_tcp_object, candidate)
        accepted = True
        reason = "accepted"
        if translation_delta > self.config.max_translation_delta_m:
            accepted, reason = False, "translation_gate"
        elif rotation_delta > self.config.max_rotation_delta_deg:
            accepted, reason = False, "rotation_gate"
        elif (
            self.config.min_backend_score is not None
            and estimate.score < self.config.min_backend_score
        ):
            accepted, reason = False, "score_gate"
        return HeldObjectRefinementUpdate(
            accepted,
            request.object_name,
            candidate if accepted else None,
            translation_delta,
            rotation_delta,
            reason,
            source=estimate.source,
            score=float(estimate.score),
            metadata={"mask_pixels": pixels, **dict(estimate.metadata)},
        )


class HeldObjectFrameSource(Protocol):
    def capture(
        self,
        object_name: str,
        current: JointConfiguration,
        prior_T_tcp_object: np.ndarray,
    ) -> HeldObjectRefinementRequest: ...


@dataclass(frozen=True)
class CachedHeldObjectFrameSource:
    """Hardware-free source for calibration checks and recorded wrist frames."""

    frame: FrameObservation
    mask: np.ndarray
    T_tcp_camera: np.ndarray

    def capture(
        self,
        object_name: str,
        current: JointConfiguration,
        prior_T_tcp_object: np.ndarray,
    ) -> HeldObjectRefinementRequest:
        del current
        return HeldObjectRefinementRequest(
            object_name=object_name,
            frame=self.frame,
            mask=self.mask,
            T_tcp_camera=self.T_tcp_camera,
            prior_T_tcp_object=prior_T_tcp_object,
        )


class LiftRefinementHook:
    """Future live-wrist adapter; tests and cached replay can supply any frame source."""

    def __init__(
        self,
        refiner: HeldObjectPoseRefiner,
        frame_source: HeldObjectFrameSource,
        initial_T_tcp_object: np.ndarray,
    ) -> None:
        self.refiner = refiner
        self.frame_source = frame_source
        self.T_tcp_object = _transform(initial_T_tcp_object, "initial_T_tcp_object")

    def refine_after_lift(
        self, object_name: str, current: JointConfiguration
    ) -> HeldObjectRefinementUpdate:
        request = self.frame_source.capture(
            object_name, current, self.T_tcp_object.copy()
        )
        if request.object_name != object_name:
            request = replace(request, object_name=object_name)
        update = self.refiner.refine(request)
        if update.accepted and update.T_tcp_object is not None:
            self.T_tcp_object = update.T_tcp_object.copy()
        return update


class HeldObjectRefinementHook(Protocol):
    def refine_after_lift(
        self, object_name: str, current: JointConfiguration
    ) -> HeldObjectRefinementUpdate: ...
