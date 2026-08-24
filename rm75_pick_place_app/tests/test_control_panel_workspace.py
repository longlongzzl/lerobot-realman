from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rm75_app.web import control_panel


class ControlPanelWorkspaceTest(unittest.TestCase):
    def test_task_ui_is_one_workspace_with_unique_dom_ids(self) -> None:
        html = control_panel.INDEX_HTML
        self.assertIn('data-tab="sceneTab">任务工作台', html)
        self.assertIn('data-tab="pickTab">感知与执行调试', html)
        self.assertNotIn('id="llmTab"', html)
        self.assertIn('id="llmPlanBtn"', html)
        self.assertIn('id="llmValidateBtn"', html)
        self.assertIn('id="llmManiskillPreviewBtn"', html)
        self.assertIn('id="maniskillPreviewDialog"', html)
        self.assertIn('id="maniskillPreviewImage"', html)
        self.assertIn("'/api/llm/maniskill-preview'", html)
        self.assertIn('id="validationStatus"', html)
        self.assertIn('id="llmLoadSceneBtn"', html)
        self.assertIn('id="virtualSceneCanvas"', html)
        self.assertIn('value="openai-compatible" selected', html)
        self.assertIn('value="qwen3.8-max"', html)
        self.assertIn('value="RM75_VLM_API_KEY"', html)
        self.assertIn('id="llmProxyUrl"', html)
        self.assertIn('value="http://127.0.0.1:7897"', html)

        ids = re.findall(r'\bid="([^"]+)"', html)
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        self.assertEqual(duplicates, [])

    def test_plan_generation_is_not_overwritten_by_stale_validation_status(self) -> None:
        html = control_panel.INDEX_HTML
        self.assertIn("data.llm_mode === 'plan'", html)
        self.assertIn("const managedLlmMode = data.llm_mode === 'validate' || data.llm_mode === 'execute'", html)

        previous_result = control_panel.latest_llm_result
        previous_mode = control_panel.llm_process_mode
        observed_modes: list[str | None] = []

        def materialize(_config, _command):
            observed_modes.append(control_panel.llm_process_mode)
            return {"generated": True}

        try:
            control_panel.llm_process_mode = "validate"
            with control_panel.app.test_client() as client, patch.object(
                control_panel.llm_process, "is_running", return_value=False
            ), patch.object(control_panel, "_materialize_llm_from_web", side_effect=materialize), patch.object(
                control_panel,
                "_llm_result_summary",
                return_value={"step_count": 1, "manifest_file": "/tmp/new_manifest.json"},
            ):
                response = client.post(
                    "/api/llm/plan",
                    json={"command": "抓起网球", "config": {}},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(observed_modes, ["plan"])
            self.assertIsNone(control_panel.llm_process_mode)
        finally:
            control_panel.latest_llm_result = previous_result
            control_panel.llm_process_mode = previous_mode

    def test_latest_validation_report_uses_newest_complete_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "task_validation" / "old" / "three_gate_report.json"
            new = root / "task_validation" / "new" / "three_gate_report.json"
            old.parent.mkdir(parents=True)
            new.parent.mkdir(parents=True)
            old.write_text(json.dumps({"plan_file": "/tmp/old.json", "passed": False}), encoding="utf-8")
            new.write_text(json.dumps({"plan_file": "/tmp/new.json", "passed": True}), encoding="utf-8")
            old_mtime = old.stat().st_mtime
            old_mtime -= 10
            os.utime(old, (old_mtime, old_mtime))

            with patch.object(control_panel, "RUNTIME_DIR", root):
                report = control_panel.latest_task_validation_report()

            self.assertIsNotNone(report)
            self.assertTrue(report["passed"])
            self.assertEqual(report["plan_file"], "/tmp/new.json")

    def test_polling_perception_summary_drops_heavy_model_payloads(self) -> None:
        compact = control_panel.compact_perception_result(
            {
                "result_path": "/tmp/result.json",
                "pose_found_objects": ["carriot"],
                "pose_details": {
                    "carriot": {
                        "score": 0.91,
                        "translation_m": [0.1, 0.2, 0.7],
                        "detection_ism": {"segmentation": {"counts": [1] * 10000}},
                    }
                },
                "pose_results": [{"segmentation": {"counts": [1] * 10000}}],
                "known_mask_results": [{"segmentation": {"counts": [1] * 10000}}],
            }
        )

        self.assertEqual(compact["pose_details"]["carriot"]["score"], 0.91)
        self.assertNotIn("pose_results", compact)
        self.assertNotIn("known_mask_results", compact)
        self.assertNotIn("detection_ism", compact["pose_details"]["carriot"])

    def test_virtual_scene_can_be_loaded_into_task_workspace(self) -> None:
        scene_file = control_panel.TEST_SCENE_DIR / "current_table.json"
        with control_panel.app.test_client() as client:
            response = client.post("/api/llm/scene/load", json={"scene_file": str(scene_file)})

        self.assertEqual(response.status_code, 200)
        scene = response.get_json()["scene"]
        self.assertEqual(scene["name"], "current_table.json")
        self.assertGreaterEqual(scene["object_count"], 2)
        self.assertTrue(any(item["object_id"] == "desk" for item in scene["objects"]))
        self.assertIn("front_axis", scene["coordinate_frame"])

    def test_workspace_execution_requires_matching_passed_report(self) -> None:
        manifest = {"manipulation_plan_file": "/tmp/current_plan.json", "steps": []}
        with control_panel.app.test_client() as client, patch.object(
            control_panel, "_llm_manifest_execution_command", return_value=(["true"], manifest)
        ), patch.object(control_panel.llm_process, "start") as start:
            with patch.object(
                control_panel,
                "latest_task_validation_report",
                return_value={"plan_file": "/tmp/old_plan.json", "passed": True},
            ):
                response = client.post("/api/llm/start", json={"manifest_file": "/tmp/manifest.json", "require_validation": True})
                self.assertEqual(response.status_code, 400)
                start.assert_not_called()

            with patch.object(
                control_panel,
                "latest_task_validation_report",
                return_value={"plan_file": "/tmp/current_plan.json", "passed": True},
            ):
                response = client.post("/api/llm/start", json={"manifest_file": "/tmp/manifest.json", "require_validation": True})
                self.assertEqual(response.status_code, 200)
                start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
