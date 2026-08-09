from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rm75_app.web.scene_workbench import SceneWorkbench


class SceneWorkbenchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bank_root = self.root / "banks"
        self.bank_root.mkdir()
        (self.bank_root / "carriot_dinov2_vits14.npz").touch()
        self.workbench = SceneWorkbench(
            self.root / "workbench",
            asset_names=["carriot", "tennis"],
            asset_dir=self.root / "assets",
            bank_root=self.bank_root,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _result(self) -> dict:
        scene = self.root / "scene"
        shared = scene / "shared_frame"
        shared.mkdir(parents=True, exist_ok=True)
        for name in ("rgb.png", "depth.png", "camera.json"):
            (shared / name).write_bytes(b"test")
        known_mask = scene / "known.png"
        unknown_mask = scene / "unknown.png"
        known_mask.write_bytes(b"known")
        unknown_mask.write_bytes(b"unknown")
        return {
            "result_path": str(scene / "result.json"),
            "scene_dir": str(scene),
            "rgb_path": str(shared / "rgb.png"),
            "pose_found_objects": ["carriot"],
            "pose_details": {"carriot": {"score": 0.91}},
            "known_mask_results": [
                {"ok": True, "object_name": "carriot", "mask_bbox": [10, 10, 100, 100], "mask_pixels": 7000, "mask_path": str(known_mask), "selected": {"model_score": 0.9}}
            ],
            "discovery_results": [
                {"ok": True, "mask_bbox": [11, 11, 99, 99], "mask_pixels": 6900, "selected": {"model_score": 0.88}},
                {"ok": True, "mask_bbox": [200, 100, 270, 180], "mask_pixels": 4300, "mask_path": str(unknown_mask), "selected": {"model_score": 0.82}},
            ],
        }

    def test_refresh_separates_known_and_unknown_and_reuses_ids(self):
        first = self.workbench.refresh(self._result())["snapshot"]
        self.assertEqual(first["counts"]["known"], 1)
        self.assertEqual(first["counts"]["unknown"], 1)
        ids = [item["instance_id"] for item in first["instances"]]
        second = self.workbench.refresh(self._result())["snapshot"]
        self.assertEqual([item["instance_id"] for item in second["instances"]], ids)

    def test_unknown_job_prepares_reconstruction_bundle(self):
        snapshot = self.workbench.refresh(self._result())["snapshot"]
        unknown = next(item for item in snapshot["instances"] if item["knownness"] == "unknown")
        result = self.workbench.create_jobs([unknown["instance_id"]], provider="observed")
        job = result["jobs"][0]
        self.assertEqual(job["status"], "capture_ready")
        self.assertTrue((Path(job["frame_dir"]) / "mask.png").exists())
        self.assertIn("openworld-geometry", job["command"])

    def test_plan_freezes_only_when_valid(self):
        snapshot = self.workbench.refresh(self._result())["snapshot"]
        known = next(item for item in snapshot["instances"] if item["knownness"] == "known")
        state = self.workbench.save_plan(
            [{"type": "pick_place", "instance_id": known["instance_id"], "destination": "slot_1"}]
        )
        self.assertTrue(state["plan"]["valid"])
        self.assertTrue(state["snapshot"]["frozen"])

    def test_unknown_cannot_be_picked_when_only_capture_is_ready(self):
        snapshot = self.workbench.refresh(self._result())["snapshot"]
        unknown = next(item for item in snapshot["instances"] if item["knownness"] == "unknown")
        self.workbench.create_jobs([unknown["instance_id"]], provider="observed")
        state = self.workbench.save_plan(
            [{"type": "pick_place", "instance_id": unknown["instance_id"], "destination": "slot_1"}]
        )
        self.assertFalse(state["plan"]["valid"])
        self.assertFalse(state["snapshot"]["frozen"])


if __name__ == "__main__":
    unittest.main()
