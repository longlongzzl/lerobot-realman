from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

from .models import SegmentationPrediction


class CutieMaskTracker:
    """Quality-gated adapter around CUTIE's public inference API.

    Prediction runs with ``end=True`` so CUTIE cannot silently write an
    unverified mask.  The RRTrack controller explicitly injects only masks that
    pass its short- or long-term gate.
    """

    def __init__(
        self,
        *,
        cutie_root: str | Path | None = None,
        weights_path: str | Path | None = None,
        device: str = "cuda",
        max_internal_size: int = 0,
        max_long_anchors: int = 6,
    ) -> None:
        root_path = None
        if cutie_root:
            root_path = Path(cutie_root).expanduser().resolve()
            root = str(root_path)
            if root not in sys.path:
                sys.path.insert(0, root)
        try:
            import torch
            from hydra import compose, initialize_config_dir
            from omegaconf import open_dict
            from cutie.inference.inference_core import InferenceCore
            from cutie.inference.utils.args_utils import get_dataset_cfg
            from cutie.model.cutie import CUTIE
        except ImportError as exc:
            raise ImportError(
                "CUTIE is required for the rrtrack entry. Install the official "
                "hkchengrex/Cutie package or pass --cutie-root."
            ) from exc

        self.torch = torch
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        if root_path is None:
            import cutie

            root_path = Path(cutie.__file__).resolve().parent.parent
        model_weights = Path(weights_path).expanduser() if weights_path else root_path / "weights" / "cutie-base-mega.pth"
        if not model_weights.exists() or model_weights.stat().st_size == 0:
            raise FileNotFoundError(
                "missing CUTIE model weight cutie-base-mega.pth; place it at "
                f"{model_weights} or pass --cutie-weights. Official release: "
                "https://github.com/hkchengrex/Cutie/releases/tag/v1.0"
            )
        config_dir = root_path / "cutie" / "config"
        with initialize_config_dir(version_base="1.3.2", config_dir=str(config_dir), job_name="rm75_rrtrack"):
            cfg = compose(config_name="eval_config", overrides=["dataset=generic"])
        with open_dict(cfg):
            cfg.weights = str(model_weights)
            if int(max_internal_size) > 0:
                cfg.max_internal_size = int(max_internal_size)
        get_dataset_cfg(cfg)
        self.model = CUTIE(cfg).to(self.device).eval()
        state = torch.load(model_weights, map_location=self.device)
        self.model.load_weights(state)
        self.processor = InferenceCore(self.model, cfg=cfg)
        self.initialized = False
        self._long_anchors: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=max(1, int(max_long_anchors)))
        self._latest_short: tuple[np.ndarray, np.ndarray] | None = None

    def _image_tensor(self, rgb: np.ndarray):
        return self.torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float().div_(255.0).to(self.device)

    def _mask_tensor(self, mask: np.ndarray):
        return self.torch.from_numpy(np.asarray(mask, dtype=np.uint8)).long().to(self.device)

    @staticmethod
    def _prediction(probability) -> SegmentationPrediction:
        probabilities = probability.detach().float().cpu().numpy()
        foreground = probabilities[1] if probabilities.shape[0] > 1 else np.zeros(probabilities.shape[1:], np.float32)
        return SegmentationPrediction(mask=foreground >= 0.5, foreground_probability=foreground)

    def initialize(self, rgb: np.ndarray, mask: np.ndarray) -> SegmentationPrediction:
        rgb_copy = np.asarray(rgb, dtype=np.uint8).copy()
        mask_copy = np.asarray(mask, dtype=bool).copy()
        with self.torch.inference_mode():
            probability = self.processor.step(
                self._image_tensor(rgb_copy),
                self._mask_tensor(mask_copy),
                objects=[1],
                idx_mask=True,
                force_permanent=True,
            )
        self.initialized = True
        self._long_anchors.clear()
        self._long_anchors.append((rgb_copy, mask_copy))
        return self._prediction(probability)

    def predict(self, rgb: np.ndarray) -> SegmentationPrediction:
        if not self.initialized:
            raise RuntimeError("CUTIE tracker is not initialized")
        # No sensory/working/long-term writeback before geometric validation.
        with self.torch.inference_mode():
            probability = self.processor.step(self._image_tensor(rgb), end=True)
        return self._prediction(probability)

    def inject(self, rgb: np.ndarray, mask: np.ndarray, *, long_term: bool) -> None:
        if not np.any(mask):
            return
        anchor = (np.asarray(rgb, dtype=np.uint8).copy(), np.asarray(mask, dtype=bool).copy())
        if long_term:
            self._long_anchors.append(anchor)
            self._rebuild_from_long_anchors()
            return
        self._latest_short = anchor
        with self.torch.inference_mode():
            self.processor.step(
                self._image_tensor(anchor[0]),
                self._mask_tensor(anchor[1]),
                objects=[1],
                idx_mask=True,
                force_permanent=False,
            )

    def _rebuild_from_long_anchors(self) -> None:
        """Rebuild a bounded permanent tier from geometry-verified anchors."""
        self.processor.clear_memory()
        with self.torch.inference_mode():
            for rgb, mask in self._long_anchors:
                self.processor.step(
                    self._image_tensor(rgb),
                    self._mask_tensor(mask),
                    objects=[1],
                    idx_mask=True,
                    force_permanent=True,
                )
            if self._latest_short is not None:
                rgb, mask = self._latest_short
                self.processor.step(
                    self._image_tensor(rgb),
                    self._mask_tensor(mask),
                    objects=[1],
                    idx_mask=True,
                    force_permanent=False,
                )

    def clear_non_permanent_memory(self) -> None:
        self.processor.clear_non_permanent_memory()
        self._latest_short = None
