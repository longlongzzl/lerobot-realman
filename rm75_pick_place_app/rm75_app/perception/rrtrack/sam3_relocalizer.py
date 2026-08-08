from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


class SAM3TextRelocalizer:
    """On-demand full-frame SAM3 proposals used only after CUTIE loses support."""

    def __init__(
        self,
        *,
        prompt: str,
        output_root: str | Path,
        python_executable: str,
        provider_script: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
        confidence: float = 0.30,
        resolution: int = 1008,
        max_candidates: int = 3,
        timeout_s: float = 180.0,
    ) -> None:
        self.prompt = str(prompt)
        self.output_root = Path(output_root).expanduser()
        self.python_executable = str(Path(python_executable).expanduser())
        self.provider_script = Path(provider_script).expanduser()
        checkpoint_text = str(checkpoint_path or "").strip()
        self.checkpoint_path = str(Path(checkpoint_text).expanduser()) if checkpoint_text else ""
        self.device = str(device)
        self.confidence = float(confidence)
        self.resolution = int(resolution)
        self.max_candidates = max(1, int(max_candidates))
        self.timeout_s = float(timeout_s)

    def __call__(self, frame) -> list[np.ndarray]:
        frame_dir = self.output_root / f"frame_{int(frame.frame_index):06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = frame_dir / "rgb.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(np.asarray(frame.rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        command = [
            self.python_executable,
            str(self.provider_script),
            "--rgb-path",
            str(rgb_path),
            "--output-dir",
            str(frame_dir),
            "--mode",
            "text",
            "--prompt",
            self.prompt,
            "--checkpoint-path",
            self.checkpoint_path,
            "--device",
            self.device,
            "--confidence-threshold",
            str(self.confidence),
            "--resolution",
            str(self.resolution),
            "--sam3-max-masks-per-item",
            str(self.max_candidates),
        ]
        proc = subprocess.run(command, text=True, capture_output=True, timeout=self.timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(f"SAM3 recovery failed: {proc.stderr[-2000:]}")
        payload_path = frame_dir / "sam3_result.json"
        with payload_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        candidates = payload.get("all_candidates", [payload])
        masks: list[np.ndarray] = []
        for candidate in candidates:
            mask = cv2.imread(str(candidate.get("mask_path", "")), cv2.IMREAD_GRAYSCALE)
            if mask is not None and np.count_nonzero(mask) > 0:
                masks.append(mask > 0)
        return masks
