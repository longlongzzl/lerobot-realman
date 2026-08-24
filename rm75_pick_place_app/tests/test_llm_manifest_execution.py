from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rm75_app.runtime.llm_manifest_execution import execute_manifest, load_execution_manifest
from rm75_app.web.control_panel import _llm_manifest_execution_command


class LlmManifestExecutionTest(unittest.TestCase):
    def test_uncombined_manifest_keeps_every_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            commands = []
            for index in (1, 2, 3):
                command = case / f"step_{index:02d}_command.sh"
                command.write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
                commands.append(command)
            manifest = case / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "output_dir": str(case),
                        "steps": [
                            {"index": index, "command_file": str(command)}
                            for index, command in enumerate(commands, start=1)
                        ],
                        "combined_command": {"available": False, "reason": "test"},
                    }
                ),
                encoding="utf-8",
            )

            _, payload = load_execution_manifest(manifest)
            command, _ = _llm_manifest_execution_command(manifest)

            self.assertEqual(len(payload["steps"]), 3)
            self.assertIn("rm75_app.runtime.llm_manifest_execution", command)
            self.assertNotEqual(command[0], "bash")

            report = execute_manifest(manifest)
            self.assertTrue(report["ok"])
            self.assertEqual(report["planned_step_count"], 3)
            self.assertEqual(report["executed_command_count"], 3)
            self.assertFalse(report["used_combined_command"])

    def test_manifest_rejects_command_outside_case_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "case"
            case.mkdir()
            outside = root / "outside.sh"
            outside.write_text("true\n", encoding="utf-8")
            manifest = case / "manifest.json"
            manifest.write_text(
                json.dumps({"steps": [{"index": 1, "command_file": str(outside)}]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "outside"):
                load_execution_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
