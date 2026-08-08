from __future__ import annotations

import cv2
import numpy as np

from .models import Agreement


def binary_entropy(probability: np.ndarray | None, mask: np.ndarray) -> float:
    if probability is None:
        return 0.0
    prob = np.asarray(probability, dtype=np.float32)
    support = np.asarray(mask, dtype=bool)
    if prob.shape != support.shape or not np.any(support):
        return 1.0
    p = np.clip(prob[support], 1e-6, 1.0 - 1e-6)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0)
    return float(np.mean(entropy))


def rendered_mask_agreement(
    observed_mask: np.ndarray,
    rendered_mask: np.ndarray,
    foreground_probability: np.ndarray | None = None,
) -> Agreement:
    observed = np.asarray(observed_mask, dtype=bool)
    rendered = np.asarray(rendered_mask, dtype=bool)
    if observed.shape != rendered.shape:
        raise ValueError(f"mask shape mismatch: observed={observed.shape}, rendered={rendered.shape}")
    intersection = observed & rendered
    observed_area = int(np.count_nonzero(observed))
    rendered_area = int(np.count_nonzero(rendered))
    intersection_area = int(np.count_nonzero(intersection))
    return Agreement(
        precision=float(intersection_area / max(observed_area, 1)),
        support=float(intersection_area / max(rendered_area, 1)),
        entropy=binary_entropy(foreground_probability, observed),
        mask_area=observed_area,
        rendered_area=rendered_area,
        intersection_area=intersection_area,
    )


class TriangleMeshMaskRenderer:
    """CPU binary silhouette renderer used for geometry gating.

    A binary silhouette does not require color/depth shading. Projecting and
    filling all front-visible mesh triangles gives a deterministic fallback
    without introducing another GPU rendering context.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)

    def render(self, T_cam_obj: np.ndarray, K: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        height, width = (int(image_shape[0]), int(image_shape[1]))
        pose = np.asarray(T_cam_obj, dtype=np.float64).reshape(4, 4)
        points = self.vertices @ pose[:3, :3].T + pose[:3, 3]
        z = points[:, 2]
        valid = z > 1e-5
        uvw = points @ np.asarray(K, dtype=np.float64).reshape(3, 3).T
        uv = np.zeros((len(points), 2), dtype=np.float64)
        uv[valid] = uvw[valid, :2] / uvw[valid, 2:3]
        output = np.zeros((height, width), dtype=np.uint8)
        if not np.any(valid):
            return output.astype(bool)
        for face in self.faces:
            if not np.all(valid[face]):
                continue
            polygon = np.rint(uv[face]).astype(np.int32)
            if (
                np.max(polygon[:, 0]) < 0
                or np.min(polygon[:, 0]) >= width
                or np.max(polygon[:, 1]) < 0
                or np.min(polygon[:, 1]) >= height
            ):
                continue
            cv2.fillConvexPoly(output, polygon, 1)
        return output.astype(bool)


def snap_translation_from_mask_depth(
    previous_pose: np.ndarray,
    mask: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_depth_px: int,
) -> tuple[np.ndarray | None, dict]:
    observed = np.asarray(mask, dtype=bool)
    ys, xs = np.where(observed)
    if xs.size == 0:
        return None, {"reason": "empty_mask"}
    valid_depth = observed & np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    values = np.asarray(depth_m, dtype=np.float64)[valid_depth]
    if values.size < int(min_valid_depth_px):
        return None, {"reason": "insufficient_depth", "valid_depth_pixels": int(values.size)}
    # RRTrack uses the 2D bounding-box center and median in-mask depth.
    u = 0.5 * (float(xs.min()) + float(xs.max()))
    v = 0.5 * (float(ys.min()) + float(ys.max()))
    z = float(np.median(values))
    xyz = np.linalg.inv(np.asarray(K, dtype=np.float64).reshape(3, 3)) @ np.asarray([u, v, 1.0])
    xyz *= z
    proposal = np.asarray(previous_pose, dtype=np.float64).reshape(4, 4).copy()
    proposal[:3, 3] = xyz
    return proposal, {"reason": "ok", "bbox_center_px": [u, v], "median_depth_m": z, "translation_m": xyz.tolist()}
