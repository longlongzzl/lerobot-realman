from __future__ import annotations

import numpy as np

from rm75_app.perception.held_object_refinement import (
    CachedHeldObjectFrameSource,
    HeldObjectPoseRefiner,
    HeldObjectRefinementConfig,
    HeldObjectRefinementRequest,
    LiftRefinementHook,
)
from rm75_app.perception.rrtrack.models import FrameObservation, PoseEstimate
from rm75_app.planning.contracts import JointConfiguration


def _frame() -> FrameObservation:
    return FrameObservation(
        rgb=np.zeros((20, 30, 3), dtype=np.uint8),
        depth_m=np.ones((20, 30), dtype=np.float32),
        K=np.asarray([[200.0, 0.0, 15.0], [0.0, 200.0, 10.0], [0.0, 0.0, 1.0]]),
    )


class FakeLocalBackend:
    def __init__(self, correction: np.ndarray) -> None:
        self.correction = correction
        self.initial_pose: np.ndarray | None = None
        self.calls = 0

    def refine(self, frame, mask, initial_pose):
        self.calls += 1
        self.initial_pose = np.asarray(initial_pose).copy()
        return PoseEstimate(
            np.asarray(initial_pose) @ self.correction,
            score=0.9,
            source="fake_foundationpose",
        )


def _request(mask: np.ndarray | None = None) -> HeldObjectRefinementRequest:
    T_tcp_camera = np.eye(4)
    T_tcp_camera[:3, 3] = [0.04, 0.0, 0.06]
    prior = np.eye(4)
    prior[:3, 3] = [0.01, -0.003, 0.10]
    return HeldObjectRefinementRequest(
        "carrot",
        _frame(),
        np.ones((20, 30), dtype=bool) if mask is None else mask,
        T_tcp_camera,
        prior,
    )


def test_refiner_predicts_camera_pose_and_returns_tcp_relative_pose() -> None:
    correction = np.eye(4)
    correction[0, 3] = 0.005
    backend = FakeLocalBackend(correction)
    request = _request()

    result = HeldObjectPoseRefiner(backend).refine(request)

    expected_initial = np.linalg.inv(request.T_tcp_camera) @ request.prior_T_tcp_object
    assert np.allclose(backend.initial_pose, expected_initial)
    assert result.accepted
    assert result.source == "fake_foundationpose"
    assert np.allclose(
        result.T_tcp_object,
        request.T_tcp_camera @ expected_initial @ correction,
    )
    assert np.isclose(result.translation_delta_m, 0.005)


def test_refiner_rejects_large_pose_jump() -> None:
    correction = np.eye(4)
    correction[1, 3] = 0.05

    result = HeldObjectPoseRefiner(FakeLocalBackend(correction)).refine(_request())

    assert not result.accepted
    assert result.reason == "translation_gate"
    assert result.T_tcp_object is None


def test_refiner_rejects_small_mask_without_running_backend() -> None:
    backend = FakeLocalBackend(np.eye(4))
    mask = np.zeros((20, 30), dtype=bool)
    mask[:2, :2] = True

    result = HeldObjectPoseRefiner(
        backend, HeldObjectRefinementConfig(min_mask_pixels=10)
    ).refine(_request(mask))

    assert not result.accepted
    assert result.reason == "mask_too_small:4"
    assert backend.calls == 0


def test_cached_source_exercises_the_same_optional_lift_hook_without_hardware() -> None:
    correction = np.eye(4)
    correction[2, 3] = -0.004
    backend = FakeLocalBackend(correction)
    request = _request()
    source = CachedHeldObjectFrameSource(
        request.frame, request.mask, request.T_tcp_camera
    )
    hook = LiftRefinementHook(
        HeldObjectPoseRefiner(backend), source, request.prior_T_tcp_object
    )

    result = hook.refine_after_lift(
        "carrot", JointConfiguration(("j1",), np.asarray([0.0]))
    )

    assert result.accepted
    assert np.allclose(hook.T_tcp_object, result.T_tcp_object)
