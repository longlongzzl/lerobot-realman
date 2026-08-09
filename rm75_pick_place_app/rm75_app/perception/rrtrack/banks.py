from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .models import DescriptorEncoder


@dataclass(frozen=True)
class TemplateCandidate:
    T_cam_obj: np.ndarray
    similarity: float
    source: str
    index: int


class FeaturePoseBank:
    def __init__(self, name: str, capacity: int | None = None) -> None:
        self.name = str(name)
        self.capacity = None if capacity is None else int(capacity)
        self._features: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._poses: deque[np.ndarray] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._features)

    def add(self, feature: np.ndarray, T_cam_obj: np.ndarray) -> None:
        normalized = _normalize_feature(feature)
        self._features.append(normalized)
        self._poses.append(np.asarray(T_cam_obj, dtype=np.float64).reshape(4, 4).copy())

    def retrieve(self, feature: np.ndarray, top_k: int) -> list[TemplateCandidate]:
        if not self._features:
            return []
        query = _normalize_feature(feature)
        matrix = np.stack(tuple(self._features), axis=0)
        similarities = matrix @ query
        order = np.argsort(similarities)[::-1][: max(1, int(top_k))]
        return [
            TemplateCandidate(
                T_cam_obj=np.asarray(tuple(self._poses)[int(index)]).copy(),
                similarity=float(similarities[int(index)]),
                source=self.name,
                index=int(index),
            )
            for index in order
        ]

    def save_npz(self, path: str | Path) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        feature_dim = self._features[0].shape[0] if self._features else 0
        features = np.stack(tuple(self._features), axis=0) if self._features else np.empty((0, feature_dim), np.float32)
        poses = np.stack(tuple(self._poses), axis=0) if self._poses else np.empty((0, 4, 4), np.float64)
        np.savez_compressed(output, features=features, poses=poses, name=self.name)
        return output

    @classmethod
    def load_npz(cls, path: str | Path, *, name: str = "offline") -> "FeaturePoseBank":
        payload = np.load(Path(path).expanduser(), allow_pickle=False)
        features = np.asarray(payload["features"], dtype=np.float32)
        poses = np.asarray(payload["poses"], dtype=np.float64)
        if features.ndim != 2 or poses.shape != (features.shape[0], 4, 4):
            raise ValueError(f"invalid RRTrack bank arrays: features={features.shape}, poses={poses.shape}")
        bank = cls(name=name)
        for feature, pose in zip(features, poses):
            bank.add(feature, pose)
        return bank


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    value = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("descriptor must have a finite non-zero norm")
    return value / norm


class DinoV2Descriptor:
    """DINOv2 CLS descriptor used by RRTrack's two recovery banks."""

    def __init__(
        self,
        *,
        model_name: str = "dinov2_vits14",
        repo_or_dir: str = "facebookresearch/dinov2",
        device: str = "cuda",
        input_size: int = 224,
        context_padding: float = 0.15,
        source: str = "github",
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.input_size = int(input_size)
        self.context_padding = float(context_padding)
        self.model = torch.hub.load(repo_or_dir, model_name, source=source).to(self.device).eval()

    def encode(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb, dtype=np.uint8)
        support = np.asarray(mask, dtype=bool)
        ys, xs = np.where(support)
        if xs.size == 0:
            raise ValueError("cannot encode an empty recovery mask")
        height, width = support.shape
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        pad_x = int(round((x2 - x1) * self.context_padding))
        pad_y = int(round((y2 - y1) * self.context_padding))
        x1, x2 = max(0, x1 - pad_x), min(width, x2 + pad_x)
        y1, y2 = max(0, y1 - pad_y), min(height, y2 + pad_y)
        masked = image.copy()
        masked[~support] = 0
        crop = cv2.resize(masked[y1:y2, x1:x2], (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        # FoundationPose changes PyTorch's global default tensor type to CUDA.
        # Keep every preprocessing tensor explicit so recovery remains valid
        # regardless of which model ran first in this process.
        tensor = self.torch.from_numpy(crop).permute(2, 0, 1).to(
            device=self.device, dtype=self.torch.float32
        ).div_(255.0)
        mean = self.torch.tensor(
            [0.485, 0.456, 0.406], device=self.device, dtype=self.torch.float32
        ).view(3, 1, 1)
        std = self.torch.tensor(
            [0.229, 0.224, 0.225], device=self.device, dtype=self.torch.float32
        ).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0)
        with self.torch.inference_mode():
            feature = self.model(tensor)
        if isinstance(feature, dict):
            feature = feature.get("x_norm_clstoken", next(iter(feature.values())))
        value = feature.detach().float().cpu().numpy().reshape(-1)
        return _normalize_feature(value)


def sam6d_render_camera_pose(raw_pose: np.ndarray) -> np.ndarray:
    """Convert SAM-6D's Blender camera pose to the renderer convention."""
    pose = np.asarray(raw_pose, dtype=np.float64).reshape(4, 4).copy()
    pose[:3, 1:3] *= -1.0
    pose[:3, 3] *= 0.002
    return pose


def build_sam6d_template_bank(
    templates_dir: str | Path,
    cam_poses_path: str | Path,
    encoder: DescriptorEncoder,
    *,
    base_views: int = 128,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> FeaturePoseBank:
    """Build RRTrack's 128-view + 180-degree augmented offline bank."""
    root = Path(templates_dir).expanduser()
    camera_poses = np.load(Path(cam_poses_path).expanduser())
    available = [
        index
        for index in range(len(camera_poses))
        if (root / f"rgb_{index}.png").exists() and (root / f"mask_{index}.png").exists()
    ]
    if not available:
        raise FileNotFoundError(f"no paired SAM-6D rgb_N.png/mask_N.png templates under {root}")
    count = min(max(1, int(base_views)), len(available))
    positions = np.linspace(0, len(available) - 1, count).round().astype(int)
    selected = [available[int(position)] for position in positions]

    bank = FeaturePoseBank("offline")
    inplane_180 = np.diag([-1.0, -1.0, 1.0])
    for sequence_index, template_index in enumerate(selected, start=1):
        bgr = cv2.imread(str(root / f"rgb_{template_index}.png"), cv2.IMREAD_COLOR)
        mask_u8 = cv2.imread(str(root / f"mask_{template_index}.png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask_u8 is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = mask_u8 > 0
        T_cam_obj = np.linalg.inv(sam6d_render_camera_pose(camera_poses[template_index]))
        bank.add(encoder.encode(rgb, mask), T_cam_obj)

        rotated_rgb = np.ascontiguousarray(np.rot90(rgb, 2))
        rotated_mask = np.ascontiguousarray(np.rot90(mask, 2))
        augmented_pose = T_cam_obj.copy()
        augmented_pose[:3, :3] = inplane_180 @ augmented_pose[:3, :3]
        bank.add(encoder.encode(rotated_rgb, rotated_mask), augmented_pose)
        if progress is not None:
            progress(sequence_index, len(selected), template_index, len(bank))
    return bank
