from __future__ import annotations

import numpy as np
import pytest
import trimesh

from rm75_app.assets.object_specs import get_object_spec, resolve_object_spec_scales
from rm75_app.orchestration.multi_object_executor import SceneObjectState, TaskSceneState
from rm75_app.pickplace.atom_task_builder import AtomTaskBuilderConfig, FixedSceneAtomTaskBuilder, _pose_to_matrix
from rm75_app.tasks.manipulation_plan import ManipulationAtom, ManipulationPrimitive


def test_builder_preserves_object_goal_for_every_grasp_pair() -> None:
    source = np.eye(4)
    source[:3, 3] = [0.35, 0.05, 0.03]
    target = np.eye(4)
    target[:3, 3] = [0.20, -0.10, 0.04]
    scene = TaskSceneState(
        {"carrot_1": SceneObjectState("carrot_1", "carriot", source)},
        joint_names=("j1", "j2"),
        joint_positions=[0.0, 0.0],
    )
    atom = ManipulationAtom(
        "atom_01",
        ManipulationPrimitive.PICK_PLACE,
        "carrot_1",
        "carriot",
        target,
    )
    builder = FixedSceneAtomTaskBuilder(
        config=AtomTaskBuilderConfig(
            grasp_yaw_offsets_deg=(0.0, 90.0),
            grasp_tilt_toward_base_deg=(0.0,),
        )
    )
    task = builder(atom, scene)
    assert len(task.grasp_candidates) == 2
    assert set(task.place_candidates_by_grasp) == {
        candidate.candidate_id for candidate in task.grasp_candidates
    }
    for grasp in task.grasp_candidates:
        place = task.places_for_grasp(grasp.candidate_id)[0]
        T_world_tcp_grasp = _pose_to_matrix(grasp.pose)
        T_tcp_object = np.linalg.inv(T_world_tcp_grasp) @ source
        T_world_tcp_place = _pose_to_matrix(place.pose)
        reconstructed_target = T_world_tcp_place @ T_tcp_object
        np.testing.assert_allclose(reconstructed_target, target, atol=1e-6)


def test_builder_converts_world_scene_to_robot_base_planning_frame() -> None:
    source = np.eye(4)
    source[:3, 3] = [-0.05, 0.12, 0.04]
    target = np.eye(4)
    target[:3, 3] = [-0.25, -0.28, 0.10]
    scene = TaskSceneState(
        {"ball": SceneObjectState("ball", "tennis", source)},
        joint_names=("j1", "j2"),
        joint_positions=[0.0, 0.0],
    )
    atom = ManipulationAtom(
        "atom_01", ManipulationPrimitive.PICK_PLACE, "ball", "tennis", target
    )
    world_task = FixedSceneAtomTaskBuilder(
        config=AtomTaskBuilderConfig(
            grasp_yaw_offsets_deg=(0.0,),
            grasp_tilt_toward_base_deg=(0.0,),
            sphere_grasp_yaw_offsets_deg=(0.0,),
        )
    )(atom, scene)
    base_task = FixedSceneAtomTaskBuilder(
        config=AtomTaskBuilderConfig(
            grasp_yaw_offsets_deg=(0.0,),
            grasp_tilt_toward_base_deg=(0.0,),
            sphere_grasp_yaw_offsets_deg=(0.0,),
            robot_base_world_xyz_m=(-0.615, 0.0, 0.0),
        )
    )(atom, scene)

    expected_delta = np.asarray([0.615, 0.0, 0.0])
    np.testing.assert_allclose(
        base_task.grasp_candidates[0].pose.position
        - world_task.grasp_candidates[0].pose.position,
        expected_delta,
    )
    np.testing.assert_allclose(
        base_task.place_candidates[0].pose.position
        - world_task.place_candidates[0].pose.position,
        expected_delta,
    )
    np.testing.assert_allclose(
        base_task.scene.objects[0].pose.position
        - world_task.scene.objects[0].pose.position,
        expected_delta,
    )


def test_builder_uses_sphere_proxy_and_automatic_attachment_for_tennis() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [0.2, -0.2, 0.04]
    scene = TaskSceneState(
        {"ball": SceneObjectState("ball", "tennis", pose)},
        joint_names=("j1", "j2"),
        joint_positions=[0.0, 0.0],
    )
    atom = ManipulationAtom(
        "atom_01",
        ManipulationPrimitive.PICK_PLACE,
        "ball",
        "tennis",
        pose,
    )

    task = FixedSceneAtomTaskBuilder()(atom, scene)

    collision = next(item for item in task.scene.objects if item.name == "ball")
    assert collision.kind == "sphere"
    assert collision.radius is not None and collision.radius > 0.0
    assert collision.metadata["collision_proxy"] == "sphere"
    assert collision.metadata["attachment_num_spheres"] is None
    assert collision.metadata["visual_mesh_path"].endswith("tennis.glb")


def test_builder_uses_oriented_cuboid_proxy_for_non_spherical_asset() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [0.2, -0.2, 0.04]
    scene = TaskSceneState(
        {"carrot": SceneObjectState("carrot", "carriot", pose)},
        joint_names=("j1", "j2"),
        joint_positions=[0.0, 0.0],
    )
    atom = ManipulationAtom(
        "atom_01",
        ManipulationPrimitive.PICK_PLACE,
        "carrot",
        "carriot",
        pose,
    )

    collision = FixedSceneAtomTaskBuilder()(atom, scene).scene.objects[0]

    assert collision.kind == "cuboid"
    assert collision.metadata["collision_proxy"] == "cuboid"
    assert collision.dimensions is not None
    assert np.max(collision.dimensions) > 4.0 * np.min(collision.dimensions)


def test_symmetric_tennis_grasp_yaw_ignores_object_rotation() -> None:
    builder = FixedSceneAtomTaskBuilder(
        config=AtomTaskBuilderConfig(
            grasp_yaw_offsets_deg=(0.0,),
            sphere_grasp_yaw_offsets_deg=(0.0,),
        )
    )
    first = np.eye(4)
    first[:3, 3] = [0.1, -0.2, 0.04]
    second = first.copy()
    second[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    first_grasp = builder._grasp_candidates("tennis", first)[0]
    second_grasp = builder._grasp_candidates("tennis", second)[0]

    np.testing.assert_allclose(
        first_grasp.pose.quaternion_wxyz,
        second_grasp.pose.quaternion_wxyz,
        atol=1e-8,
    )


def test_default_grasp_relation_sets_add_pen_refinement_candidates() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [0.25, 0.10, 0.04]
    builder = FixedSceneAtomTaskBuilder()

    tennis = builder._grasp_candidates("tennis", transform)
    carrot = builder._grasp_candidates("carriot", transform)
    pen = builder._grasp_candidates("bi", transform)

    assert len(tennis) == 64
    assert len(carrot) == 16
    coarse_pen = [
        item for item in pen if item.metadata.get("refinement_parent_id") is None
    ]
    refined_pen = [
        item for item in pen if item.metadata.get("refinement_parent_id") is not None
    ]
    assert len(coarse_pen) == 56
    assert len(refined_pen) == 128
    assert {item.metadata["search_tier"] for item in refined_pen} == {2}
    assert {
        item.metadata["yaw_offset_deg"] for item in tennis
    } == set(builder.config.sphere_grasp_yaw_offsets_deg)
    assert {
        item.metadata["tilt_toward_base_deg"] for item in tennis
    } == set(builder.config.sphere_grasp_tilt_toward_base_deg)
    assert {
        round(float(item.metadata["axis_shift_m"]), 3) for item in coarse_pen
    } == {0.0, -0.048, -0.04, -0.02, 0.02, 0.04, 0.048}
    assert {
        item.metadata["tilt_toward_base_deg"] for item in coarse_pen
    } == {0.0, 20.0}
    assert {
        item.metadata["tilt_toward_base_deg"] for item in carrot
    } == set(builder.config.grasp_tilt_toward_base_deg)


def test_axial_objects_expand_equivalent_place_spin_without_changing_axis() -> None:
    source = np.eye(4)
    source[:3, 3] = [0.25, 0.10, 0.04]
    target = np.eye(4)
    target[:3, 3] = [0.10, -0.20, 0.12]
    scene = TaskSceneState(
        {"pen": SceneObjectState("pen", "bi", source)},
        joint_names=("j1", "j2"),
        joint_positions=[0.0, 0.0],
    )
    atom = ManipulationAtom(
        "atom_01", ManipulationPrimitive.PICK_PLACE, "pen", "bi", target
    )

    task = FixedSceneAtomTaskBuilder()(atom, scene)
    places = task.places_for_grasp(task.grasp_candidates[0].candidate_id)

    assert len(places) == 8
    assert {item.metadata["axial_spin_deg"] for item in places} == {
        0.0,
        -90.0,
        90.0,
        180.0,
    }
    assert {item.metadata["axis_flipped"] for item in places} == {False, True}
    assert {
        (item.metadata["axis_flipped"], item.metadata["axial_spin_deg"]):
        item.metadata["search_tier"]
        for item in places
    } == {
        (False, 0.0): 0,
        (False, -90.0): 1,
        (False, 90.0): 1,
        (False, 180.0): 2,
        (True, 0.0): 3,
        (True, -90.0): 3,
        (True, 90.0): 3,
        (True, 180.0): 3,
    }
    expected_axis = target[:3, :3] @ np.asarray([0.0, 1.0, 0.0])
    for place in places:
        equivalent = np.asarray(place.metadata["target_object_pose"])
        actual_axis = equivalent[:3, :3] @ np.asarray([0.0, 1.0, 0.0])
        expected_sign = -1.0 if place.metadata["axis_flipped"] else 1.0
        np.testing.assert_allclose(
            actual_axis, expected_sign * expected_axis, atol=1e-8
        )


def test_grasp_position_uses_world_mesh_center_not_asset_origin() -> None:
    builder = FixedSceneAtomTaskBuilder(
        config=AtomTaskBuilderConfig(
            grasp_yaw_offsets_deg=(0.0,),
            grasp_tilt_toward_base_deg=(0.0,),
            grasp_z_offset_m=0.005,
        )
    )
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    transform[:3, 3] = [-0.2, 0.1, 0.03]

    grasp = builder._grasp_candidates("carriot", transform)[0]

    spec = get_object_spec("carriot")
    assert spec is not None
    mesh_scale, _ = resolve_object_spec_scales(spec)
    mesh = trimesh.load(spec.mesh_file, force="scene", process=False)
    scale = np.eye(4)
    scale[:3, :3] *= mesh_scale
    mesh.apply_transform(transform @ scale)
    expected = np.mean(np.asarray(mesh.bounds), axis=0)
    expected[2] += 0.005

    np.testing.assert_allclose(grasp.pose.position, expected, atol=1e-8)


def test_sphere_release_expands_base_tilt_candidates_and_preserves_target_pose() -> None:
    source = np.eye(4)
    source[:3, 3] = [0.0, 0.14, 0.037]
    target = np.eye(4)
    target[:3, 3] = [-0.30, -0.28, 0.103]
    scene = TaskSceneState(
        {"ball": SceneObjectState("ball", "tennis", source)},
        joint_names=("j1", "j2"),
        joint_positions=[0.0, 0.0],
    )
    atom = ManipulationAtom(
        "atom_01",
        ManipulationPrimitive.PICK_PLACE,
        "ball",
        "tennis",
        target,
    )
    builder = FixedSceneAtomTaskBuilder(
        config=AtomTaskBuilderConfig(
            grasp_yaw_offsets_deg=(0.0,),
            sphere_grasp_yaw_offsets_deg=(0.0,),
        )
    )

    task = builder(atom, scene)
    places = task.places_for_grasp("grasp_00_+0deg")

    assert len(places) == 16
    assert {item.metadata["release_tilt_toward_base_deg"] for item in places} == {
        0.0,
        15.0,
        30.0,
        45.0,
    }
    for place in places:
        assert place.metadata["sphere_release"] is True
        assert place.metadata["sphere_release_lift_m"] == pytest.approx(0.03)
        np.testing.assert_allclose(place.metadata["target_object_pose"], target)
    tilted = next(
        item
        for item in places
        if item.metadata["release_tilt_toward_base_deg"] == 15.0
        and item.metadata["release_roll_deg"] == 0.0
    )
    tilted_rotation = _pose_to_matrix(tilted.pose)[:3, :3]
    assert tilted_rotation[2, 2] > -1.0
    assert tilted_rotation[2, 2] == pytest.approx(-np.cos(np.deg2rad(15.0)))
