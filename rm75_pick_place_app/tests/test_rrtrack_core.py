from __future__ import annotations

import unittest

import numpy as np
import trimesh
from PIL import Image

from rm75_app.assets.object_specs import OBJECT_SPECS
from rm75_app.perception.rrtrack.agreement import rendered_mask_agreement, snap_translation_from_mask_depth
from rm75_app.perception.rrtrack.banks import FeaturePoseBank
from rm75_app.perception.rrtrack.config import RRTrackConfig
from rm75_app.perception.rrtrack.foundationpose_adapter import foundationpose_compatible_mesh
from rm75_app.perception.rrtrack.models import FrameObservation, PoseEstimate, SegmentationPrediction, TrackerState
from rm75_app.perception.rrtrack.scene_registry import build_scene_registry, resolve_active_result
from rm75_app.perception.rrtrack.tracker import RRTracker
from rm75_app.runtime.rrtrack_build_all_banks import _geometry_groups


class FakeSegmenter:
    def __init__(self, masks):
        self.masks = list(masks)
        self.injected = []
        self.clears = 0

    def initialize(self, rgb, mask):
        return SegmentationPrediction(np.asarray(mask, bool), np.asarray(mask, np.float32) * 0.98)

    def predict(self, rgb):
        mask = np.asarray(self.masks.pop(0), bool)
        return SegmentationPrediction(mask, mask.astype(np.float32) * 0.98)

    def inject(self, rgb, mask, *, long_term):
        self.injected.append((bool(long_term), int(np.count_nonzero(mask))))

    def clear_non_permanent_memory(self):
        self.clears += 1


class FakeRefiner:
    def refine(self, frame, mask, initial_pose):
        return PoseEstimate(np.asarray(initial_pose).copy(), source="fake_refine")

    def global_register(self, frame, mask):
        return PoseEstimate(np.eye(4), source="fake_global")


class FakeRenderer:
    def __init__(self, mask):
        self.mask = np.asarray(mask, bool)

    def render(self, T_cam_obj, K, image_shape):
        return self.mask.copy()


class FakeDescriptor:
    def encode(self, rgb, mask):
        return np.asarray([1.0, 0.0, 0.0], np.float32)


def make_frame(index=0):
    return FrameObservation(
        np.zeros((20, 30, 3), np.uint8),
        np.ones((20, 30), np.float32),
        np.asarray([[100.0, 0.0, 15.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]),
        index,
    )


class RRTrackCoreTest(unittest.TestCase):
    def test_all_object_bank_groups_cover_specs_and_deduplicate_shared_meshes(self):
        groups = _geometry_groups(OBJECT_SPECS)
        covered = sorted(name for group in groups for name in group["object_names"])
        self.assertEqual(covered, sorted(OBJECT_SPECS))
        self.assertLess(len(groups), len(OBJECT_SPECS))
        triangle = next(group for group in groups if "red_triangle_front" in group["object_names"])
        self.assertEqual(len(triangle["object_names"]), 4)
        pillars = next(group for group in groups if "tingzi_pillar_front_left" in group["object_names"])
        self.assertEqual(len(pillars["object_names"]), 4)

    def test_pbr_texture_is_normalized_for_foundationpose(self):
        mesh = trimesh.creation.box()
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.new("RGB", (4, 4), (255, 128, 0))
            ),
        )
        compatible = foundationpose_compatible_mesh(mesh)
        self.assertIsInstance(compatible.visual.material, trimesh.visual.material.SimpleMaterial)
        self.assertIsNotNone(compatible.visual.material.image)

    def test_paper_agreement_equation(self):
        observed = np.zeros((5, 5), bool)
        rendered = np.zeros((5, 5), bool)
        observed[1:3, 1:3] = True
        rendered[2:4, 1:3] = True
        result = rendered_mask_agreement(observed, rendered)
        self.assertAlmostEqual(result.precision, 0.5)
        self.assertAlmostEqual(result.support, 0.5)

    def test_snap_keeps_rotation_and_replaces_translation(self):
        pose = np.eye(4)
        pose[:3, :3] = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        mask = np.zeros((20, 30), bool)
        mask[8:13, 13:18] = True
        proposal, info = snap_translation_from_mask_depth(
            pose,
            mask,
            np.ones(mask.shape, np.float32),
            make_frame().K,
            min_depth_m=0.1,
            max_depth_m=2.0,
            min_valid_depth_px=4,
        )
        self.assertEqual(info["reason"], "ok")
        np.testing.assert_allclose(proposal[:3, :3], pose[:3, :3])
        np.testing.assert_allclose(proposal[:3, 3], [0.0, 0.0, 1.0], atol=1e-6)

    def test_loss_then_dual_bank_style_recovery(self):
        good = np.zeros((20, 30), bool)
        good[5:15, 10:20] = True
        empty = np.zeros_like(good)
        segmenter = FakeSegmenter([empty, good])
        bank = FeaturePoseBank("offline")
        bank.add(np.asarray([1.0, 0.0, 0.0]), np.eye(4))
        config = RRTrackConfig(
            min_track_area_px=8,
            min_snap_area_px=8,
            min_valid_depth_px=4,
            lost_patience_frames=1,
            recovery_retry_interval=1,
            precision_track=0.5,
            precision_floor=0.2,
        )
        tracker = RRTracker(
            segmenter=segmenter,
            pose_refiner=FakeRefiner(),
            renderer=FakeRenderer(good),
            descriptor=FakeDescriptor(),
            offline_bank=bank,
            config=config,
        )
        tracker.initialize(make_frame(0), good, np.eye(4))
        lost = tracker.step(make_frame(1))
        self.assertEqual(lost.state, TrackerState.LOST)
        recovered = tracker.step(make_frame(2))
        self.assertEqual(recovered.state, TrackerState.TRACKING)
        self.assertEqual(recovered.event, "recovered")
        self.assertTrue(recovered.accepted)
        self.assertGreaterEqual(segmenter.clears, 1)

    def test_sam3_multi_candidate_reappearance_is_geometry_checked(self):
        good = np.zeros((20, 30), bool)
        good[5:15, 10:20] = True
        wrong = np.zeros_like(good)
        wrong[1:5, 1:5] = True
        empty = np.zeros_like(good)
        segmenter = FakeSegmenter([empty, empty])
        config = RRTrackConfig(
            min_track_area_px=8,
            min_snap_area_px=8,
            min_valid_depth_px=4,
            lost_patience_frames=1,
            recovery_retry_interval=1,
            sam3_recovery_after_frames=1,
            precision_track=0.5,
            precision_floor=0.2,
        )
        tracker = RRTracker(
            segmenter=segmenter,
            pose_refiner=FakeRefiner(),
            renderer=FakeRenderer(good),
            config=config,
            relocalize_mask=lambda frame: [wrong, good],
        )
        tracker.initialize(make_frame(0), good, np.eye(4))
        self.assertEqual(tracker.step(make_frame(1)).state, TrackerState.LOST)
        recovered = tracker.step(make_frame(2))
        self.assertEqual(recovered.event, "recovered")
        self.assertEqual(recovered.metadata["mask_candidate_count"], 2)
        np.testing.assert_array_equal(recovered.mask, good)

    def test_scene_registry_preserves_all_instances_and_active_handoff(self):
        payload = {
            "results": [
                {"object_name": "redcube", "ok": True, "score": 0.8, "T_cam_obj": np.eye(4).tolist(), "run_dir": "/tmp/a"},
                {"object_name": "redcube", "ok": True, "score": 0.7, "T_cam_obj": np.eye(4).tolist(), "run_dir": "/tmp/b"},
                {"object_name": "tennis", "ok": True, "score": 0.9, "T_cam_obj": np.eye(4).tolist(), "run_dir": "/tmp/c"},
            ]
        }
        registry = build_scene_registry(payload, "redcube", 1)
        self.assertEqual(registry["active_instance_id"], "redcube:1")
        self.assertEqual(len(registry["objects"]), 3)
        self.assertEqual(resolve_active_result(payload, "redcube", 1)["run_dir"], "/tmp/b")


if __name__ == "__main__":
    unittest.main()
