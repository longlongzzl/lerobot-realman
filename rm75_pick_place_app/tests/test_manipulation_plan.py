from __future__ import annotations

import pytest

from rm75_app.tasks.manipulation_plan import (
    ManipulationPlan,
    ManipulationPrimitive,
    compile_resolved_steps,
    load_plan,
)


IDENTITY = [
    [1.0, 0.0, 0.0, 0.1],
    [0.0, 1.0, 0.0, 0.2],
    [0.0, 0.0, 1.0, 0.3],
    [0.0, 0.0, 0.0, 1.0],
]


def _step(**overrides):
    value = {
        "source_id": "pen_1",
        "source_spec": "bi",
        "target_object_id": "cup_1",
        "target_pose": IDENTITY,
        "operator": "inside",
        "primitive": "insert_vertical",
        "place_mode": "auto",
    }
    value.update(overrides)
    return value


def test_compile_expands_backend_names_into_public_task_atoms(tmp_path) -> None:
    plan = compile_resolved_steps(
        plan_id="demo",
        scene_file=tmp_path / "scene.json",
        steps=[
            _step(),
            _step(
                source_id="ball_1",
                source_spec="tennis",
                operator="desk_slot",
                primitive="place_on_slots",
                place_mode="surface_place",
            ),
        ],
        user_command="把笔插进笔筒，再放网球",
    )
    assert plan.atoms[0].primitive is ManipulationPrimitive.INSERT
    assert plan.atoms[0].success.relation == "inside"
    assert plan.atoms[1].primitive is ManipulationPrimitive.PICK_PLACE
    assert plan.atoms[1].success.relation == "slot"
    assert plan.atoms[1].depends_on == ("atom_01",)
    assert plan.as_dict()["schema_version"] == "rm75.manipulation_plan/v1"


def test_plan_rejects_forward_dependencies(tmp_path) -> None:
    atom = compile_resolved_steps(
        plan_id="source",
        scene_file=tmp_path / "scene.json",
        steps=[_step()],
    ).atoms[0]
    broken = atom.__class__(**{**atom.__dict__, "depends_on": ("later",)})
    with pytest.raises(ValueError, match="dependencies"):
        ManipulationPlan("broken", str(tmp_path / "scene.json"), (broken,))


def test_pose_must_be_homogeneous(tmp_path) -> None:
    bad_pose = [row[:] for row in IDENTITY]
    bad_pose[3][3] = 0.0
    with pytest.raises(ValueError, match="homogeneous"):
        compile_resolved_steps(
            plan_id="bad",
            scene_file=tmp_path / "scene.json",
            steps=[_step(target_pose=bad_pose)],
        )


def test_plan_json_round_trip(tmp_path) -> None:
    import json

    plan = compile_resolved_steps(
        plan_id="round_trip",
        scene_file=tmp_path / "scene.json",
        steps=[_step(semantic_step={"operator": "inside", "target_ref": "cup_1"})],
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.as_dict()), encoding="utf-8")
    loaded = load_plan(path)
    assert loaded == plan
    assert loaded.atoms[0].metadata["semantic_step"]["target_ref"] == "cup_1"
