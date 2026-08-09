from __future__ import annotations

from collections import deque
import time
from typing import Callable

import numpy as np

from .adaptive import EMAMADThresholdManager
from .agreement import rendered_mask_agreement, snap_translation_from_mask_depth
from .banks import FeaturePoseBank, TemplateCandidate
from .config import RRTrackConfig
from .models import (
    Agreement,
    DescriptorEncoder,
    FrameObservation,
    MaskRenderer,
    MaskTracker,
    PoseEstimate,
    PoseRefiner,
    RRTrackOutput,
    SegmentationPrediction,
    TrackerState,
)


class RRTracker:
    """Paper-derived closed-loop 2D--6D tracker with recoverable state."""

    def __init__(
        self,
        *,
        segmenter: MaskTracker,
        pose_refiner: PoseRefiner,
        renderer: MaskRenderer,
        config: RRTrackConfig | None = None,
        descriptor: DescriptorEncoder | None = None,
        offline_bank: FeaturePoseBank | None = None,
        online_bank: FeaturePoseBank | None = None,
        relocalize_mask: Callable[[FrameObservation], np.ndarray | list[np.ndarray] | None] | None = None,
    ) -> None:
        self.config = config or RRTrackConfig()
        self.segmenter = segmenter
        self.pose_refiner = pose_refiner
        self.renderer = renderer
        self.descriptor = descriptor
        self.offline_bank = offline_bank or FeaturePoseBank("offline")
        self.online_bank = online_bank or FeaturePoseBank("online", capacity=self.config.online_bank_capacity)
        self.relocalize_mask = relocalize_mask
        self.thresholds = EMAMADThresholdManager(
            alpha=self.config.ema_alpha,
            window=self.config.mad_window,
            min_samples=self.config.adaptive_min_samples,
        )
        self.state = TrackerState.INITIALIZING
        self.pose: np.ndarray | None = None
        self.frame_index = -1
        self.lost_streak = 0
        self.last_long_memory_frame = -10**9
        self.last_bank_frame = -10**9
        self.last_recovery_attempt = -10**9
        self.last_relocalization_attempt = -10**9
        self.recovery_frames = 0
        self.retrieval_failure_streak = 0
        self.pose_history: deque[np.ndarray] = deque(maxlen=self.config.stagnation_window + 1)

    def initialize(
        self,
        frame: FrameObservation,
        mask: np.ndarray,
        T_cam_obj: np.ndarray,
    ) -> RRTrackOutput:
        observed = np.asarray(mask, dtype=bool)
        if np.count_nonzero(observed) < self.config.min_track_area_px:
            raise ValueError("initial RRTrack mask is too small")
        self.pose = np.asarray(T_cam_obj, dtype=np.float64).reshape(4, 4).copy()
        prediction = self.segmenter.initialize(frame.rgb, observed)
        rendered = self.renderer.render(self.pose, frame.K, frame.depth_m.shape)
        agreement = rendered_mask_agreement(observed, rendered, prediction.foreground_probability)
        self.state = TrackerState.TRACKING
        self.frame_index = int(frame.frame_index)
        self.pose_history.clear()
        self.pose_history.append(self.pose.copy())
        self.recovery_frames = 0
        self.retrieval_failure_streak = 0
        self._observe_stable(agreement)
        bank_updated = self._maybe_update_online_bank(frame, observed, agreement, force=True)
        return RRTrackOutput(
            frame_index=self.frame_index,
            state=self.state,
            T_cam_obj=self.pose.copy(),
            mask=observed,
            rendered_mask=rendered,
            agreement=agreement,
            accepted=True,
            event="initialized",
            memory_updates=("long",) + (("online_bank",) if bank_updated else ()),
        )

    def step(self, frame: FrameObservation) -> RRTrackOutput:
        if self.pose is None:
            raise RuntimeError("RRTracker.initialize must be called before step")
        self.frame_index = int(frame.frame_index)
        prediction = self.segmenter.predict(frame.rgb)
        observed = np.asarray(prediction.mask, dtype=bool)

        if self.state in {TrackerState.LOST, TrackerState.RECOVERING}:
            return self._step_recovery(frame, prediction)

        event = "tracked"
        metadata: dict = {}
        try:
            estimate = self.pose_refiner.refine(frame, observed, self.pose)
            candidate_pose = np.asarray(estimate.T_cam_obj, dtype=np.float64).reshape(4, 4)
            rendered = self.renderer.render(candidate_pose, frame.K, frame.depth_m.shape)
            agreement = rendered_mask_agreement(observed, rendered, prediction.foreground_probability)
        except Exception as exc:
            return self._mark_lost(frame, prediction, f"local_refiner_error:{type(exc).__name__}", {"error": str(exc)})

        if agreement.precision < self._precision_track() and agreement.mask_area >= self.config.min_snap_area_px:
            snapped_pose, snap_info = snap_translation_from_mask_depth(
                self.pose,
                observed,
                frame.depth_m,
                frame.K,
                min_depth_m=self.config.min_depth_m,
                max_depth_m=self.config.max_depth_m,
                min_valid_depth_px=self.config.min_valid_depth_px,
            )
            metadata["snap"] = snap_info
            if snapped_pose is not None:
                self.state = TrackerState.CORRECTING
                try:
                    snapped = self.pose_refiner.refine(frame, observed, snapped_pose)
                    snapped_pose = np.asarray(snapped.T_cam_obj, dtype=np.float64).reshape(4, 4)
                    snapped_render = self.renderer.render(snapped_pose, frame.K, frame.depth_m.shape)
                    snapped_agreement = rendered_mask_agreement(observed, snapped_render, prediction.foreground_probability)
                    if (snapped_agreement.precision + snapped_agreement.support) >= (agreement.precision + agreement.support):
                        candidate_pose, rendered, agreement = snapped_pose, snapped_render, snapped_agreement
                        event = "snap_corrected"
                except Exception as exc:
                    metadata["snap_refiner_error"] = str(exc)

        lost_now = (
            agreement.precision < self._precision_floor()
            or agreement.mask_area < self.config.min_track_area_px
        )
        stagnated = self._is_stagnated(candidate_pose, agreement.precision)
        self.lost_streak = self.lost_streak + 1 if (lost_now or stagnated) else 0
        if self.lost_streak >= self.config.lost_patience_frames:
            reason = "pose_stagnation" if stagnated else "mask_or_agreement_loss"
            return self._mark_lost(frame, prediction, reason, metadata, agreement=agreement, rendered=rendered)

        self.pose = candidate_pose
        self.pose_history.append(self.pose.copy())
        self.state = TrackerState.TRACKING
        updates = self._quality_gated_writeback(frame, prediction, rendered, agreement)
        if agreement.precision >= self._precision_track():
            self._observe_stable(agreement)
        return RRTrackOutput(
            frame_index=self.frame_index,
            state=self.state,
            T_cam_obj=self.pose.copy(),
            mask=observed,
            rendered_mask=rendered,
            agreement=agreement,
            accepted=agreement.precision >= self._precision_floor(),
            event=event,
            memory_updates=updates,
            metadata=metadata,
        )

    def _step_recovery(self, frame: FrameObservation, prediction: SegmentationPrediction) -> RRTrackOutput:
        observed = np.asarray(prediction.mask, dtype=bool)
        self.recovery_frames += 1
        candidate_masks = [observed] if np.count_nonzero(observed) >= self.config.min_track_area_px else []
        propagated_candidate_count = len(candidate_masks)
        relocalizer_error = None
        if (
            self.relocalize_mask is not None
            and self.recovery_frames >= self.config.sam3_recovery_after_frames
            and self.frame_index - self.last_relocalization_attempt >= self.config.recovery_retry_interval
        ):
            self.last_relocalization_attempt = self.frame_index
            try:
                recovered = self.relocalize_mask(frame)
                recovered_masks = recovered if isinstance(recovered, list) else ([] if recovered is None else [recovered])
                candidate_masks.extend(
                    np.asarray(mask, dtype=bool)
                    for mask in recovered_masks
                    if np.count_nonzero(mask) >= self.config.min_track_area_px
                )
            except Exception as exc:
                relocalizer_error = str(exc)

        blank = Agreement(0.0, 0.0, 1.0, int(np.count_nonzero(observed)), 0, 0)
        if not candidate_masks:
            self.state = TrackerState.LOST
            return RRTrackOutput(
                self.frame_index,
                self.state,
                self.pose.copy(),
                observed,
                None,
                blank,
                False,
                "waiting_for_reappearance",
                metadata={"relocalizer_error": relocalizer_error} if relocalizer_error else {},
            )
        if self.frame_index - self.last_recovery_attempt < self.config.recovery_retry_interval:
            return RRTrackOutput(
                self.frame_index,
                TrackerState.LOST,
                self.pose.copy(),
                observed,
                None,
                blank,
                False,
                "recovery_backoff",
            )

        self.last_recovery_attempt = self.frame_index
        self.state = TrackerState.RECOVERING
        attempts = []
        for candidate_index, candidate_mask in enumerate(candidate_masks):
            candidate_prediction = SegmentationPrediction(
                candidate_mask,
                prediction.foreground_probability if candidate_index < propagated_candidate_count else None,
            )
            result = self._recover_pose(frame, candidate_mask, candidate_prediction)
            attempts.append((result, candidate_mask, candidate_prediction))
        (estimate, agreement, rendered, source, recovery_meta), observed, prediction = max(
            attempts,
            key=lambda item: (
                item[0][0] is not None and item[0][1].precision >= self._precision_track(),
                float(item[0][0].score) if item[0][0] is not None else -1e9,
                item[0][1].precision,
            ),
        )
        recovery_meta["mask_candidate_count"] = len(candidate_masks)
        recovery_meta["propagated_mask_candidates"] = propagated_candidate_count
        recovery_meta["sam3_mask_candidates"] = len(candidate_masks) - propagated_candidate_count
        if relocalizer_error:
            recovery_meta["relocalizer_error"] = relocalizer_error
        if estimate is None or agreement.precision < self._precision_track():
            self.state = TrackerState.LOST
            return RRTrackOutput(
                self.frame_index,
                self.state,
                self.pose.copy(),
                observed,
                rendered,
                agreement,
                False,
                "recovery_rejected",
                recovery_source=source,
                metadata=recovery_meta,
            )

        self.pose = np.asarray(estimate.T_cam_obj, dtype=np.float64).reshape(4, 4)
        self.pose_history.clear()
        self.pose_history.append(self.pose.copy())
        self.lost_streak = 0
        self.recovery_frames = 0
        self.retrieval_failure_streak = 0
        self.state = TrackerState.TRACKING
        self.segmenter.clear_non_permanent_memory()
        self.segmenter.inject(frame.rgb, rendered, long_term=True)
        self.segmenter.inject(frame.rgb, observed, long_term=False)
        self.last_long_memory_frame = self.frame_index
        bank_updated = self._maybe_update_online_bank(frame, observed, agreement, force=True)
        self._observe_stable(agreement)
        updates = ("memory_reset", "long", "short") + (("online_bank",) if bank_updated else ())
        return RRTrackOutput(
            self.frame_index,
            self.state,
            self.pose.copy(),
            observed,
            rendered,
            agreement,
            True,
            "recovered",
            updates,
            source,
            recovery_meta,
        )

    def _recover_pose(
        self,
        frame: FrameObservation,
        mask: np.ndarray,
        prediction: SegmentationPrediction,
    ) -> tuple[PoseEstimate | None, Agreement, np.ndarray | None, str | None, dict]:
        candidates: list[TemplateCandidate] = []
        metadata: dict = {"retrieved": [], "timing_ms": {}}
        feature = None
        if self.descriptor is not None:
            try:
                started = time.perf_counter()
                feature = self.descriptor.encode(frame.rgb, mask)
                metadata["timing_ms"]["descriptor"] = (time.perf_counter() - started) * 1000.0
                for bank, threshold in (
                    (self.offline_bank, self.config.similarity_offline),
                    (self.online_bank, self.config.similarity_online),
                ):
                    threshold = self.thresholds.lower_bound(
                        f"similarity_{bank.name}",
                        threshold,
                        mad_scale=self.config.adaptive_mad_scale,
                    )
                    for candidate in bank.retrieve(feature, self.config.retrieval_top_k):
                        metadata["retrieved"].append(
                            {"source": candidate.source, "index": candidate.index, "similarity": candidate.similarity}
                        )
                        if candidate.similarity >= threshold:
                            candidates.append(candidate)
            except Exception as exc:
                metadata["descriptor_error"] = str(exc)

        proposals: list[np.ndarray] = []
        proposal_candidates: list[TemplateCandidate] = []
        for candidate in candidates:
            proposal, _ = snap_translation_from_mask_depth(
                candidate.T_cam_obj,
                mask,
                frame.depth_m,
                frame.K,
                min_depth_m=self.config.min_depth_m,
                max_depth_m=self.config.max_depth_m,
                min_valid_depth_px=self.config.min_valid_depth_px,
            )
            if proposal is None:
                continue
            proposals.append(proposal)
            proposal_candidates.append(candidate)

        ranked_estimates: list[tuple[PoseEstimate, TemplateCandidate]] = []
        refine_many = getattr(self.pose_refiner, "refine_candidates", None)
        if proposals and callable(refine_many):
            try:
                started = time.perf_counter()
                estimates = refine_many(frame, mask, proposals)
                metadata["timing_ms"]["candidate_batch_refine"] = (time.perf_counter() - started) * 1000.0
                ranked_estimates = list(zip(estimates, proposal_candidates))
            except Exception as exc:
                metadata["candidate_batch_refine_error"] = str(exc)
        if not ranked_estimates:
            for proposal, candidate in zip(proposals, proposal_candidates):
                try:
                    ranked_estimates.append((self.pose_refiner.refine(frame, mask, proposal), candidate))
                except Exception:
                    continue

        if ranked_estimates:
            # RRTrack first selects FoundationPose's highest-ranker hypothesis,
            # then accepts/rejects that single winner using rendered-mask P.
            has_ranker_scores = any(estimate.score != 0.0 for estimate, _ in ranked_estimates)
            if has_ranker_scores:
                estimate, candidate = max(ranked_estimates, key=lambda item: float(item[0].score))
            else:
                scored = []
                for estimate, candidate in ranked_estimates:
                    rendered = self.renderer.render(estimate.T_cam_obj, frame.K, frame.depth_m.shape)
                    agreement = rendered_mask_agreement(mask, rendered, prediction.foreground_probability)
                    scored.append((agreement.precision + 0.25 * agreement.support, estimate, candidate, agreement, rendered))
                _, estimate, candidate, agreement, rendered = max(scored, key=lambda item: item[0])
                metadata["ranker_fallback"] = "mask_agreement"
            if has_ranker_scores:
                rendered = self.renderer.render(estimate.T_cam_obj, frame.K, frame.depth_m.shape)
                agreement = rendered_mask_agreement(mask, rendered, prediction.foreground_probability)
            metadata["selected_candidate"] = {
                "source": candidate.source,
                "index": candidate.index,
                "similarity": candidate.similarity,
                "foundationpose_ranker_score": float(estimate.score),
            }
            if agreement.precision >= self._precision_track():
                self.retrieval_failure_streak = 0
                self.thresholds.observe(f"similarity_{candidate.source}", candidate.similarity)
                return estimate, agreement, rendered, candidate.source, metadata

            self.retrieval_failure_streak += 1
            metadata["retrieval_failure_streak"] = self.retrieval_failure_streak
            if self.retrieval_failure_streak < self.config.global_register_after_retrieval_failures:
                return estimate, agreement, rendered, f"{candidate.source}_rejected", metadata
        else:
            self.retrieval_failure_streak += 1
            metadata["retrieval_failure_streak"] = self.retrieval_failure_streak
            if (
                self.descriptor is not None
                and self.retrieval_failure_streak < self.config.global_register_after_retrieval_failures
            ):
                return None, Agreement(0.0, 0.0, 1.0, int(np.count_nonzero(mask)), 0, 0), None, "retrieval_deferred", metadata

        # Paper fallback: full-sphere pose search when retrieval has no accepted candidate.
        try:
            started = time.perf_counter()
            estimate = self.pose_refiner.global_register(frame, mask)
            metadata["timing_ms"]["global_register"] = (time.perf_counter() - started) * 1000.0
            if estimate is not None:
                rendered = self.renderer.render(estimate.T_cam_obj, frame.K, frame.depth_m.shape)
                agreement = rendered_mask_agreement(mask, rendered, prediction.foreground_probability)
                return estimate, agreement, rendered, estimate.source, metadata
        except Exception as exc:
            metadata["global_register_error"] = str(exc)
        return None, Agreement(0.0, 0.0, 1.0, int(np.count_nonzero(mask)), 0, 0), None, None, metadata

    def _mark_lost(
        self,
        frame: FrameObservation,
        prediction: SegmentationPrediction,
        event: str,
        metadata: dict,
        *,
        agreement: Agreement | None = None,
        rendered: np.ndarray | None = None,
    ) -> RRTrackOutput:
        if self.state is not TrackerState.LOST:
            self.recovery_frames = 0
            self.retrieval_failure_streak = 0
        self.state = TrackerState.LOST
        observed = np.asarray(prediction.mask, dtype=bool)
        agreement = agreement or Agreement(0.0, 0.0, 1.0, int(np.count_nonzero(observed)), 0, 0)
        return RRTrackOutput(
            self.frame_index,
            self.state,
            self.pose.copy(),
            observed,
            rendered,
            agreement,
            False,
            event,
            metadata=metadata,
        )

    def _quality_gated_writeback(
        self,
        frame: FrameObservation,
        prediction: SegmentationPrediction,
        rendered: np.ndarray,
        agreement: Agreement,
    ) -> tuple[str, ...]:
        updates: list[str] = []
        if (
            agreement.precision > self._precision_long()
            and agreement.support > self._support_long()
            and self.frame_index - self.last_long_memory_frame >= self.config.long_memory_interval
        ):
            self.segmenter.inject(frame.rgb, rendered, long_term=True)
            self.last_long_memory_frame = self.frame_index
            updates.append("long")
        if agreement.precision > self._precision_short() and agreement.entropy < self._entropy_max():
            self.segmenter.inject(frame.rgb, prediction.mask, long_term=False)
            updates.append("short")
        if self._maybe_update_online_bank(frame, prediction.mask, agreement):
            updates.append("online_bank")
        return tuple(updates)

    def _maybe_update_online_bank(
        self,
        frame: FrameObservation,
        mask: np.ndarray,
        agreement: Agreement,
        *,
        force: bool = False,
    ) -> bool:
        if self.descriptor is None or self.pose is None:
            return False
        if not force and (
            agreement.precision <= self._precision_bank()
            or agreement.support <= self._support_bank()
            or self.frame_index - self.last_bank_frame < self.config.online_bank_interval
        ):
            return False
        try:
            self.online_bank.add(self.descriptor.encode(frame.rgb, mask), self.pose)
        except Exception:
            return False
        self.last_bank_frame = self.frame_index
        return True

    def _is_stagnated(self, candidate_pose: np.ndarray, precision: float) -> bool:
        if len(self.pose_history) < self.config.stagnation_window:
            return False
        old = self.pose_history[0]
        current = np.asarray(candidate_pose, dtype=np.float64).reshape(4, 4)
        relative = old[:3, :3].T @ current[:3, :3]
        cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        rotation_deg = float(np.degrees(np.arccos(cosine)))
        translation_m = float(np.linalg.norm(current[:3, 3] - old[:3, 3]))
        normalized_motion = max(
            rotation_deg / max(self.config.stagnation_rotation_deg, 1e-9),
            translation_m / max(self.config.stagnation_translation_m, 1e-9),
        )
        return normalized_motion < 1.0 and precision < self._precision_stagnation()

    def _observe_stable(self, agreement: Agreement) -> None:
        self.thresholds.observe("precision", agreement.precision)
        self.thresholds.observe("support", agreement.support)
        self.thresholds.observe("entropy", agreement.entropy)

    def _lower(self, metric: str, fallback: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return self.thresholds.lower_bound(
            metric,
            fallback,
            mad_scale=self.config.adaptive_mad_scale,
            minimum=minimum,
            maximum=maximum,
        )

    def _precision_track(self) -> float:
        return self._lower("precision", self.config.precision_track, self.config.precision_floor, 0.95)

    def _precision_floor(self) -> float:
        return min(self.config.precision_floor, self._precision_track() - 0.05)

    def _precision_stagnation(self) -> float:
        return min(self.config.precision_stagnation, self._precision_track())

    def _precision_short(self) -> float:
        return max(self.config.precision_short_memory, self._lower("precision", self.config.precision_short_memory))

    def _precision_long(self) -> float:
        return max(self.config.precision_long_memory, self._lower("precision", self.config.precision_long_memory))

    def _support_long(self) -> float:
        return max(self.config.support_long_memory, self._lower("support", self.config.support_long_memory))

    def _precision_bank(self) -> float:
        return max(self.config.precision_bank, self._lower("precision", self.config.precision_bank))

    def _support_bank(self) -> float:
        return max(self.config.support_bank, self._lower("support", self.config.support_bank))

    def _entropy_max(self) -> float:
        adaptive = self.thresholds.upper_bound(
            "entropy",
            self.config.entropy_max,
            mad_scale=self.config.adaptive_mad_scale,
        )
        return min(self.config.entropy_max, adaptive)
