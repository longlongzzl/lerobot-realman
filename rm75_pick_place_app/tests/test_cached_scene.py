from __future__ import annotations

from pathlib import Path

import numpy as np

from rm75_app.pickplace.cached_scene import load_cached_known_object_scene, topdown_grasp_candidates


def test_cached_carrot_perception_becomes_planning_scene() -> None:
    root = Path(__file__).parents[1]
    result = root / "runtime_data/rrtrack_manual_init/20260808_203311_carriot_pid1169281/sam6d_pose_result.json"
    extrinsic = root / "assets/calibration/camera_extrinsic_opencv.npy"
    scene = load_cached_known_object_scene(result, extrinsic)

    assert scene.object_name == "carriot"
    assert scene.rgb_path.is_file() and scene.depth_path.is_file() and scene.mask_path.is_file()
    assert scene.T_base_object.shape == (4, 4)
    assert scene.T_base_object[2, 3] > 0.0
    assert [item.name for item in scene.scene.objects] == ["virtual_table_plane", "carriot"]
    assert scene.object_collision.kind == "cuboid"
    assert Path(scene.object_collision.metadata["visual_mesh_path"]).name == "carriot.glb"
    assert np.all(np.asarray(scene.object_collision.dimensions) > 0.0)

    candidates = topdown_grasp_candidates(scene)
    assert len(candidates) == 4
    assert len({item.candidate_id for item in candidates}) == 4
    for candidate in candidates:
        np.testing.assert_allclose(np.linalg.norm(candidate.pose.quaternion_wxyz), 1.0)
