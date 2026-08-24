from __future__ import annotations

import json

import numpy as np

from rm75_app.tasks.manipulation_plan import ManipulationAtom, ManipulationPlan, ManipulationPrimitive
from rm75_app.validation.geometry_gate import GeometryGate
from rm75_app.validation.contracts import GateResult, GateStatus
from rm75_app.validation.three_gate import run_three_gates


def transform(x, y, z):
    value = np.eye(4)
    value[:3, 3] = [x, y, z]
    return value


def scene_file(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(
        json.dumps(
            {
                "objects": {
                    "carrot_1": {"name": "carriot", "T_world_obj": transform(0.2, 0, 0.05).tolist()},
                    "ball_1": {"name": "tennis", "T_world_obj": transform(0.5, 0, 0.05).tolist()},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_geometry_gate_accepts_clear_sequential_targets(tmp_path) -> None:
    path = scene_file(tmp_path)
    atom = ManipulationAtom("atom_01", ManipulationPrimitive.PICK_PLACE, "carrot_1", "carriot", transform(0.25, 0.25, 0.05))
    result = GeometryGate().run(ManipulationPlan("clear", str(path), (atom,)))
    assert result.passed
    assert result.checks[0]["collisions"] == []


def test_geometry_gate_rejects_obvious_object_overlap(tmp_path) -> None:
    path = scene_file(tmp_path)
    atom = ManipulationAtom("atom_01", ManipulationPrimitive.PICK_PLACE, "carrot_1", "carriot", transform(0.5, 0, 0.05))
    result = GeometryGate().run(ManipulationPlan("blocked", str(path), (atom,)))
    assert not result.passed
    assert result.checks[0]["collisions"][0]["other_object_id"] == "ball_1"
    assert result.checks[0]["collisions"][0]["severity"] == "error"


def test_three_gate_pipeline_stops_after_geometry_failure(tmp_path, monkeypatch) -> None:
    path = scene_file(tmp_path)
    atom = ManipulationAtom("atom_01", ManipulationPrimitive.PICK_PLACE, "carrot_1", "carriot", transform(0.5, 0, 0.05))
    plan = ManipulationPlan("blocked", str(path), (atom,))
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan.as_dict()), encoding="utf-8")
    monkeypatch.setattr(
        "rm75_app.validation.three_gate._run_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("GPU worker must not run")),
    )
    report = run_three_gates(plan_file, tmp_path / "report", through="maniskill")
    assert [item["status"] for item in report["gates"]] == ["failed", "skipped", "skipped"]


def test_three_gate_pipeline_passes_worker_artifacts_forward(tmp_path, monkeypatch) -> None:
    path = scene_file(tmp_path)
    atom = ManipulationAtom("atom_01", ManipulationPrimitive.PICK_PLACE, "carrot_1", "carriot", transform(0.25, 0.25, 0.05))
    plan = ManipulationPlan("clear", str(path), (atom,))
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan.as_dict()), encoding="utf-8")
    execution = tmp_path / "execution.json"
    execution.write_text("[]", encoding="utf-8")
    calls = []

    def worker(gate, *args, execution_file=None, **kwargs):
        calls.append((gate, execution_file))
        artifacts = {"execution_file": str(execution)} if gate == "curobo2" else {}
        return GateResult(gate, GateStatus.PASSED, "ok", artifacts=artifacts)

    monkeypatch.setattr("rm75_app.validation.three_gate._run_worker", worker)
    report = run_three_gates(plan_file, tmp_path / "report", through="maniskill")
    assert report["passed"]
    assert calls == [("curobo2", None), ("maniskill", execution)]
