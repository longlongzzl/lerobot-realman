from __future__ import annotations

import json

import numpy as np
import pytest

from rm75_app.runtime.maniskill_scene_preview import write_preview_scene_spec
from rm75_app.tasks.manipulation_plan import ManipulationAtom, ManipulationPlan, ManipulationPrimitive


def test_preview_scene_contains_fixed_initial_object_and_target_ghost(tmp_path) -> None:
    source = np.eye(4)
    source[:3, 3] = [-0.05, 0.12, 0.04]
    target = np.eye(4)
    target[:3, 3] = [-0.25, -0.28, 0.10]
    scene_file = tmp_path / "scene.json"
    scene_file.write_text(
        json.dumps({"objects": {"ball": {"name": "tennis", "T_world_obj": source.tolist()}}}),
        encoding="utf-8",
    )
    plan = ManipulationPlan(
        "preview",
        str(scene_file),
        (
            ManipulationAtom(
                "atom_01",
                ManipulationPrimitive.PICK_PLACE,
                "ball",
                "tennis",
                target,
                semantic_operator="inside",
            ),
        ),
    )
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan.as_dict()), encoding="utf-8")

    preview_file = write_preview_scene_spec(plan_file, tmp_path / "preview.json")
    payload = json.loads(preview_file.read_text(encoding="utf-8"))

    assert payload["objects"][0]["object_id"] == "ball"
    assert payload["objects"][0]["fixed"] is True
    assert payload["preview_views"] is True
    assert payload["targets"][0]["atom_id"] == "atom_01"
    assert payload["targets"][0]["object_id"] == "ball"
    np.testing.assert_allclose(payload["targets"][0]["pose"], target)
    assert len(payload["collision_proxies"]) == 1
    proxy = payload["collision_proxies"][0]
    assert proxy["object_id"] == "ball"
    # The planner converts its analytic sphere to the conservative OBB path.
    assert proxy["source_kind"] == "sphere"
    assert proxy["kind"] == "cuboid"
    assert proxy["dimensions"][0] == pytest.approx(proxy["dimensions"][1])
    assert payload["robot_collision_models"] == []
    np.testing.assert_allclose(
        np.asarray(payload["robot_base_world_pose"])[:3, 3], [-0.615, 0.0, 0.0]
    )
    assert payload["robot_base_world_pose_source"] == "rm75_default"
    grasp = payload["grasp_previews"][0]
    assert grasp["candidate_id"] == "grasp_00_+0deg"
    assert grasp["approach_axis"] == "world_z"
    assert grasp["distance_m"] == pytest.approx(0.08)
    pregrasp = np.asarray(grasp["pregrasp_pose"])
    grasp_pose = np.asarray(grasp["grasp_pose"])
    np.testing.assert_allclose(
        grasp_pose[:3, 3] - pregrasp[:3, 3], [0.0, 0.0, -0.08], atol=1e-8
    )


def test_preview_scene_embeds_exported_robot_collision_models(tmp_path) -> None:
    source = np.eye(4)
    source[:3, 3] = [-0.05, 0.12, 0.04]
    scene_file = tmp_path / "scene.json"
    scene_file.write_text(
        json.dumps({"objects": {"ball": {"name": "tennis", "T_world_obj": source.tolist()}}}),
        encoding="utf-8",
    )
    plan = ManipulationPlan(
        "preview",
        str(scene_file),
        (ManipulationAtom("atom_01", ManipulationPrimitive.PICK_PLACE, "ball", "tennis", source),),
    )
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan.as_dict()), encoding="utf-8")
    geometry_file = tmp_path / "robot_geometry.json"
    geometry_file.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "role": "pregrasp",
                        "spheres": [
                            {"link_name": "link_1", "center": [0.0, 0.0, 0.2], "radius": 0.05}
                        ],
                    }
                ],
                "sphere_coordinate_frame": "robot_base",
                "missing_ik_roles": ["grasp"],
            }
        ),
        encoding="utf-8",
    )

    preview_file = write_preview_scene_spec(
        plan_file,
        tmp_path / "preview.json",
        robot_geometry_file=geometry_file,
    )
    payload = json.loads(preview_file.read_text(encoding="utf-8"))

    assert payload["robot_collision_models"][0]["role"] == "pregrasp"
    assert payload["robot_collision_models"][0]["spheres"][0]["link_name"] == "link_1"
    assert payload["robot_collision_models"][0]["coordinate_frame"] == "robot_base"
    np.testing.assert_allclose(
        np.asarray(payload["robot_collision_models"][0]["pose"])[:3, 3],
        [-0.615, 0.0, 0.0],
    )
    assert payload["collision_geometry_missing_ik_roles"] == ["grasp"]


def test_preview_scene_uses_explicit_robot_base_pose(tmp_path) -> None:
    source = np.eye(4)
    scene_file = tmp_path / "scene.json"
    scene_file.write_text(
        json.dumps(
            {
                "robot_base_world_xyz_m": [-0.72, 0.03, 0.01],
                "objects": {
                    "ball": {"name": "tennis", "T_world_obj": source.tolist()}
                },
            }
        ),
        encoding="utf-8",
    )
    plan = ManipulationPlan(
        "preview",
        str(scene_file),
        (
            ManipulationAtom(
                "atom_01", ManipulationPrimitive.PICK_PLACE, "ball", "tennis", source
            ),
        ),
    )
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan.as_dict()), encoding="utf-8")

    preview_file = write_preview_scene_spec(plan_file, tmp_path / "preview.json")
    payload = json.loads(preview_file.read_text(encoding="utf-8"))

    np.testing.assert_allclose(
        np.asarray(payload["robot_base_world_pose"])[:3, 3], [-0.72, 0.03, 0.01]
    )
    assert payload["robot_base_world_pose_source"] == "robot_base_world_xyz_m"
