from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from rm75_app.tasks.manipulation_plan import load_plan
from rm75_app.validation.contracts import GateResult, GateStatus
from rm75_app.validation.geometry_gate import GeometryGate


GATE_ORDER = ("geometry", "curobo2", "maniskill")
DEFAULT_CUROBO2_PYTHON = "/home/zhangzhao/anaconda3/envs/curobo2/bin/python"
DEFAULT_MANISKILL_PYTHON = "/home/zhangzhao/anaconda3/envs/foundationpose310/bin/python"


def run_three_gates(
    plan_file: str | Path,
    output_dir: str | Path,
    *,
    through: str = "maniskill",
    render_mode: str = "rgb_array",
) -> dict:
    aliases = {"geometry": 0, "curobo": 1, "curobo2": 1, "maniskill": 2, "physics": 2}
    if through not in aliases:
        raise ValueError(f"unknown final gate {through!r}")
    final_index = aliases[through]
    plan_path = Path(plan_file).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = load_plan(plan_path)
    results: list[GateResult] = []

    geometry = GeometryGate().run(plan)
    results.append(geometry)
    _write_report(output, plan_path, results)
    if final_index >= 1:
        if geometry.passed:
            results.append(
                _run_worker(
                    "curobo2",
                    plan_path,
                    output / "gate_2_curobo2",
                    python=os.environ.get("RM75_CUROBO2_PYTHON", DEFAULT_CUROBO2_PYTHON),
                )
            )
        else:
            results.append(GateResult("curobo2", GateStatus.SKIPPED, "skipped because geometry gate failed"))
        _write_report(output, plan_path, results)
    if final_index >= 2:
        if results[-1].passed:
            execution_file = Path(results[-1].artifacts["execution_file"])
            results.append(
                _run_worker(
                    "maniskill",
                    plan_path,
                    output / "gate_3_maniskill",
                    python=os.environ.get("RM75_MANISKILL_PYTHON", DEFAULT_MANISKILL_PYTHON),
                    execution_file=execution_file,
                    render_mode=render_mode,
                )
            )
        else:
            results.append(GateResult("maniskill", GateStatus.SKIPPED, "skipped because Curobo2 gate failed"))
    return _write_report(output, plan_path, results)


def _run_worker(
    gate: str,
    plan_file: Path,
    output: Path,
    *,
    python: str,
    execution_file: Path | None = None,
    render_mode: str = "rgb_array",
) -> GateResult:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-m",
        "rm75_app.runtime.task_validation_worker",
        "--gate",
        gate,
        "--plan",
        str(plan_file),
        "--output-dir",
        str(output),
        "--render-mode",
        render_mode,
    ]
    if execution_file is not None:
        command += ["--execution-file", str(execution_file)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    result_file = output / "gate_result.json"
    if result_file.is_file():
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        return GateResult(
            payload["gate"],
            GateStatus(payload["status"]),
            payload["summary"],
            tuple(payload.get("checks") or ()),
            payload.get("artifacts") or {},
            payload.get("error"),
        )
    error = (completed.stderr or completed.stdout or f"worker exited {completed.returncode}").strip()
    return GateResult(gate, GateStatus.ERROR, f"{gate} worker crashed", error=error[-4000:])


def _write_report(output: Path, plan_file: Path, results: list[GateResult]) -> dict:
    complete = len(results) > 0 and all(item.passed for item in results)
    report = {
        "schema_version": "rm75.three_gate_report/v1",
        "created_at": time.time(),
        "plan_file": str(plan_file),
        "passed": complete,
        "gates": [item.as_dict() for item in results],
    }
    path = output / "three_gate_report.json"
    report["report_file"] = str(path)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report
