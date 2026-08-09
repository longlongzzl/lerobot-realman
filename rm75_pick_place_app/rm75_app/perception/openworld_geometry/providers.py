from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import trimesh

from .models import CompletionResult, GeometryFrame
from .pointcloud import masked_depth_to_points, voxel_reduce


class GeometryCompletionProvider(Protocol):
    def complete(self, frame: GeometryFrame, output_dir: Path) -> CompletionResult: ...


class ObservedSurfaceProvider:
    """No-hallucination fallback that starts from the measured target surface."""

    def __init__(self, voxel_size_m: float = 0.003) -> None:
        self.voxel_size_m = float(voxel_size_m)

    def complete(self, frame: GeometryFrame, output_dir: Path) -> CompletionResult:
        points, colors = masked_depth_to_points(frame)
        points, colors, _ = voxel_reduce(points, colors, 1.0, self.voxel_size_m)
        return CompletionResult(
            points_model=points,
            confidence=np.ones(len(points), np.float32),
            colors=colors,
            source="observed_surface",
            metadata={"generated": False},
        )


class RaySt3RProvider:
    """Isolated adapter around the official RaySt3R inference entrypoint."""

    def __init__(
        self,
        *,
        root: str | Path,
        python_executable: str,
        confidence_threshold: float = 5.0,
        predicted_views_per_axis: int = 3,
        filter_all_masks: bool = True,
        timeout_s: float = 300.0,
        voxel_size_m: float = 0.003,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.python_executable = str(Path(python_executable).expanduser())
        self.confidence_threshold = float(confidence_threshold)
        self.predicted_views_per_axis = int(predicted_views_per_axis)
        self.filter_all_masks = bool(filter_all_masks)
        self.timeout_s = float(timeout_s)
        self.voxel_size_m = float(voxel_size_m)
        if not (self.root / "eval_wrapper" / "eval.py").exists():
            raise FileNotFoundError(f"invalid RaySt3R root: {self.root}")

    @staticmethod
    def _write_input(frame: GeometryFrame, input_dir: Path) -> None:
        import torch

        input_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(input_dir / "rgb.png"), cv2.cvtColor(np.asarray(frame.rgb, np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(input_dir / "mask.png"), np.asarray(frame.mask, np.uint8) * 255)
        depth = np.asarray(frame.depth_m, dtype=np.float64)
        depth_u16 = np.clip(np.rint(depth / 10.0 * np.iinfo(np.uint16).max), 0, np.iinfo(np.uint16).max).astype(
            np.uint16
        )
        cv2.imwrite(str(input_dir / "depth.png"), depth_u16)
        torch.save(torch.as_tensor(np.asarray(frame.K), dtype=torch.float32), input_dir / "intrinsics.pt")
        torch.save(torch.eye(4, dtype=torch.float32), input_dir / "cam2world.pt")

    def complete(self, frame: GeometryFrame, output_dir: Path) -> CompletionResult:
        run_dir = Path(output_dir).expanduser() / "rayst3r"
        self._write_input(frame, run_dir)
        command = [
            self.python_executable,
            "eval_wrapper/eval.py",
            str(run_dir),
            "--set_conf",
            str(self.confidence_threshold),
            "--n_pred_views",
            str(self.predicted_views_per_axis),
        ]
        if self.filter_all_masks:
            command.append("--filter_all_masks")
        proc = subprocess.run(
            command,
            cwd=str(self.root),
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
        )
        (run_dir / "adapter_process.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "returncode": int(proc.returncode),
                    "stdout_tail": (proc.stdout or "")[-4000:],
                    "stderr_tail": (proc.stderr or "")[-4000:],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if proc.returncode != 0:
            raise RuntimeError(f"RaySt3R inference failed:\n{(proc.stderr or proc.stdout)[-3000:]}")
        cloud_path = run_dir / "inference_points.ply"
        if not cloud_path.exists():
            raise FileNotFoundError(f"RaySt3R produced no point cloud: {cloud_path}")
        loaded = trimesh.load(cloud_path, process=False)
        points = np.asarray(getattr(loaded, "vertices", []), dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            raise RuntimeError("RaySt3R returned an empty point cloud")
        colors = None
        visual = getattr(loaded, "visual", None)
        vertex_colors = getattr(visual, "vertex_colors", None)
        if vertex_colors is not None and len(vertex_colors) == len(points):
            colors = np.asarray(vertex_colors, dtype=np.float64)[:, :3]
        points, colors, weights = voxel_reduce(points, colors, 1.0, self.voxel_size_m)
        return CompletionResult(
            points_model=points,
            confidence=np.ones(len(points), np.float32),
            colors=colors,
            source="rayst3r",
            metadata={
                "cloud_path": str(cloud_path),
                "confidence_threshold": self.confidence_threshold,
                "predicted_views_per_axis": self.predicted_views_per_axis,
                "filtered_point_count": int(len(points)),
                "weights_sum": float(np.sum(weights)),
            },
        )
