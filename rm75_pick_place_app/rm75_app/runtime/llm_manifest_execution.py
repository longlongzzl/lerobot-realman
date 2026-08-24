"""Execute every step in a materialized LLM manifest.

This is the process boundary used by the web task workbench.  A manifest may
contain one combined Curobo2 command or a dependency-ordered list of commands.
The latter must never silently degrade to executing only its first step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rm75_app.llm.orchestrator import run_materialized_commands


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_execution_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
        raise ValueError("LLM manifest has no steps")
    if not payload["steps"]:
        raise ValueError("LLM manifest has an empty plan")

    case_root = manifest_path.parent
    command_files = [step.get("command_file") for step in payload["steps"]]
    combined = payload.get("combined_command")
    if isinstance(combined, dict) and combined.get("available"):
        command_files.append(combined.get("command_file"))
    for raw in command_files:
        command_path = Path(str(raw or "")).expanduser().resolve()
        if not raw or not command_path.is_file() or not _inside(command_path, case_root):
            raise ValueError(f"manifest command is missing or outside its case directory: {raw}")

    payload["output_dir"] = str(case_root)
    return manifest_path, payload


def execute_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path, payload = load_execution_manifest(path)
    run_results = run_materialized_commands(payload, stop_on_failure=True)
    report = {
        "ok": bool(payload.get("run_ok")),
        "manifest_file": str(manifest_path),
        "planned_step_count": len(payload["steps"]),
        "executed_command_count": len(run_results),
        "used_combined_command": bool(run_results and run_results[0].get("combined")),
        "run_results": run_results,
    }
    report_path = manifest_path.parent / "web_execution_report.json"
    report["report_file"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Execute all commands in an LLM task manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    report = execute_manifest(args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
