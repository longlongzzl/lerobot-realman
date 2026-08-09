from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import trimesh

from .models import FrameObservation, PoseEstimate


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def foundationpose_compatible_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize modern GLB PBR textures for FoundationPose's legacy loader."""
    compatible = mesh.copy()
    visual = compatible.visual
    if not isinstance(visual, trimesh.visual.texture.TextureVisuals):
        return compatible
    material = visual.material
    if getattr(material, "image", None) is not None:
        return compatible
    image = getattr(material, "baseColorTexture", None)
    if image is not None:
        compatible.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.asarray(visual.uv).copy(),
            material=trimesh.visual.material.SimpleMaterial(image=image),
        )
        return compatible
    compatible.visual = visual.to_color()
    return compatible


class FoundationPoseRefiner:
    """Frozen FoundationPose local refiner with candidate re-initialization."""

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        *,
        foundationpose_root: str | Path,
        refine_iterations: int = 2,
        register_iterations: int = 5,
        debug_dir: str | Path = "/tmp/rm75_rrtrack_foundationpose",
    ) -> None:
        root = Path(foundationpose_root).expanduser().resolve()
        if not (root / "estimater.py").exists():
            raise FileNotFoundError(f"invalid FoundationPose root: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        estimater = _load_module("rm75_rrtrack_foundationpose_estimater", root / "estimater.py")
        self._module = estimater
        glctx = estimater.dr.RasterizeCudaContext()
        mesh = foundationpose_compatible_mesh(mesh)
        self.estimator = estimater.FoundationPose(
            model_pts=np.asarray(mesh.vertices),
            model_normals=np.asarray(mesh.vertex_normals),
            mesh=mesh,
            debug=0,
            debug_dir=str(Path(debug_dir).expanduser()),
            glctx=glctx,
        )
        self.refine_iterations = int(refine_iterations)
        self.register_iterations = int(register_iterations)

    def _set_initial_pose(self, T_cam_obj: np.ndarray) -> None:
        # FoundationPose stores the pose of its centered mesh internally.
        to_center = self.estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
        centered_pose = np.asarray(T_cam_obj, dtype=np.float64).reshape(4, 4) @ np.linalg.inv(to_center)
        import torch

        self.estimator.pose_last = torch.as_tensor(centered_pose, dtype=torch.float32, device="cuda").reshape(1, 4, 4)

    def refine(self, frame: FrameObservation, mask: np.ndarray, initial_pose: np.ndarray) -> PoseEstimate:
        self._set_initial_pose(initial_pose)
        pose = self.estimator.track_one(
            rgb=np.asarray(frame.rgb),
            depth=np.asarray(frame.depth_m, dtype=np.float32),
            K=np.asarray(frame.K, dtype=np.float32),
            iteration=self.refine_iterations,
        )
        return PoseEstimate(np.asarray(pose, dtype=np.float64), source="foundationpose_refine")

    def refine_candidates(
        self,
        frame: FrameObservation,
        mask: np.ndarray,
        initial_poses: list[np.ndarray],
    ) -> list[PoseEstimate]:
        """Refine a recovery batch once and rank it with FoundationPose's scorer."""
        if not initial_poses:
            return []
        import torch

        to_center = self.estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
        centered = np.stack(
            [np.asarray(pose, dtype=np.float64).reshape(4, 4) @ np.linalg.inv(to_center) for pose in initial_poses],
            axis=0,
        )
        depth = torch.as_tensor(frame.depth_m, device="cuda", dtype=torch.float32)
        depth = self._module.erode_depth(depth, radius=2, device="cuda")
        depth = self._module.bilateral_filter_depth(depth, radius=2, device="cuda")
        xyz_map = self._module.depth2xyzmap_batch(
            depth[None],
            torch.as_tensor(frame.K, dtype=torch.float32, device="cuda")[None],
            zfar=np.inf,
        )[0]
        refined, _ = self.estimator.refiner.predict(
            mesh=self.estimator.mesh,
            mesh_tensors=self.estimator.mesh_tensors,
            rgb=np.asarray(frame.rgb),
            depth=depth,
            K=np.asarray(frame.K, dtype=np.float32),
            ob_in_cams=centered,
            normal_map=None,
            xyz_map=xyz_map,
            glctx=self.estimator.glctx,
            mesh_diameter=self.estimator.diameter,
            iteration=self.refine_iterations,
            get_vis=False,
        )
        scores, _ = self.estimator.scorer.predict(
            mesh=self.estimator.mesh,
            rgb=np.asarray(frame.rgb),
            depth=depth,
            K=np.asarray(frame.K, dtype=np.float32),
            ob_in_cams=refined.detach().cpu().numpy(),
            normal_map=None,
            mesh_tensors=self.estimator.mesh_tensors,
            glctx=self.estimator.glctx,
            mesh_diameter=self.estimator.diameter,
            get_vis=False,
        )
        refined_np = refined.detach().cpu().numpy()
        scores_np = torch.as_tensor(scores).detach().float().cpu().numpy().reshape(-1)
        return [
            PoseEstimate(
                T_cam_obj=np.asarray(pose @ to_center, dtype=np.float64),
                score=float(score),
                source="foundationpose_recovery_ranker",
                metadata={"candidate_index": int(index), "ranker_score": float(score)},
            )
            for index, (pose, score) in enumerate(zip(refined_np, scores_np))
        ]

    def global_register(self, frame: FrameObservation, mask: np.ndarray) -> PoseEstimate | None:
        if not np.any(mask):
            return None
        pose = self.estimator.register(
            K=np.asarray(frame.K, dtype=np.float32),
            rgb=np.asarray(frame.rgb),
            depth=np.asarray(frame.depth_m, dtype=np.float32),
            ob_mask=np.asarray(mask, dtype=np.uint8),
            iteration=self.register_iterations,
        )
        score = 0.0
        try:
            score = float(self.estimator.scores[0].detach().cpu().item())
        except Exception:
            pass
        return PoseEstimate(np.asarray(pose, dtype=np.float64), score=score, source="foundationpose_spherical")
