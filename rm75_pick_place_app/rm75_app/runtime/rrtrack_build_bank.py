#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rm75_app.perception.rrtrack.banks import DinoV2Descriptor, build_sam6d_template_bank


DEFAULT_CAM_POSES = (
    "/home/zhangzhao/PycharmProjects/SAM-6D/SAM-6D/"
    "Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy"
)
DEFAULT_DINOV2_ROOT = "/home/zhangzhao/.cache/torch/hub/facebookresearch_dinov2_main"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an RRTrack offline DINOv2 recovery bank from available SAM6D views")
    parser.add_argument("--templates-dir", required=True, help="SAM-6D templates containing rgb_N.png and mask_N.png")
    parser.add_argument("--cam-poses", default=DEFAULT_CAM_POSES)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dinov2-root", default=DEFAULT_DINOV2_ROOT)
    parser.add_argument("--dino-model-name", default="dinov2_vits14")
    parser.add_argument(
        "--base-views",
        type=int,
        default=128,
        help="Maximum base views; each gets 180-degree augmentation (standard 42-view cache gives 84 entries)",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dino_root = Path(args.dinov2_root).expanduser()
    encoder = DinoV2Descriptor(
        model_name=args.dino_model_name,
        repo_or_dir=str(dino_root) if dino_root.exists() else "facebookresearch/dinov2",
        source="local" if dino_root.exists() else "github",
        device=args.device,
        input_size=224,
        context_padding=0.15,
    )
    bank = build_sam6d_template_bank(
        args.templates_dir,
        args.cam_poses,
        encoder,
        base_views=args.base_views,
        progress=lambda done, total, index, entries: print(
            f"[rrtrack-bank] {done}/{total} template={index} entries={entries}"
        ),
    )
    output = bank.save_npz(args.output)
    print(f"[rrtrack-bank] saved {len(bank)} entries: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
