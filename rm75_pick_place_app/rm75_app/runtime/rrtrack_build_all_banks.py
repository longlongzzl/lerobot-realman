#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from rm75_app.assets.object_specs import OBJECT_SPECS, normalize_object_name, resolve_object_spec_scales
from rm75_app.paths import RUNTIME_DIR
from rm75_app.perception.rrtrack.banks import DinoV2Descriptor, FeaturePoseBank, build_sam6d_template_bank
from rm75_app.perception.sam6d_pose_provider import export_cad_mm, prepare_templates
from rm75_app.runtime.rrtrack_build_bank import DEFAULT_CAM_POSES, DEFAULT_DINOV2_ROOT


DEFAULT_SAM6D_ROOT = "/home/zhangzhao/PycharmProjects/SAM-6D/SAM-6D"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable RRTrack DINO banks for every known ObjectSpec")
    parser.add_argument("--object-names", nargs="+", help="Subset of canonical ObjectSpec names; default is all")
    parser.add_argument("--sam6d-root", default=DEFAULT_SAM6D_ROOT)
    parser.add_argument("--template-cache-root", default=str(RUNTIME_DIR / "sam6d_template_cache"))
    parser.add_argument("--work-root", default=str(RUNTIME_DIR / "rrtrack_bank_build_work"))
    parser.add_argument("--output-root", default=str(RUNTIME_DIR / "rrtrack_banks"))
    parser.add_argument("--cam-poses", default=DEFAULT_CAM_POSES)
    parser.add_argument("--dinov2-root", default=DEFAULT_DINOV2_ROOT)
    parser.add_argument("--dino-model-name", default="dinov2_vits14")
    parser.add_argument("--base-views", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-banks", action="store_true")
    return parser.parse_args(argv)


def _selected_specs(names: list[str] | None):
    if not names:
        return dict(OBJECT_SPECS)
    selected = {}
    for raw_name in names:
        name = normalize_object_name(raw_name)
        if name not in OBJECT_SPECS:
            raise KeyError(f"unknown ObjectSpec {raw_name!r}")
        selected[name] = OBJECT_SPECS[name]
    return selected


def _geometry_groups(specs):
    groups: dict[tuple[str, float], dict] = {}
    for name, spec in specs.items():
        mesh_scale, _ = resolve_object_spec_scales(spec)
        mesh_path = Path(spec.mesh_file).expanduser().resolve()
        key = (str(mesh_path), round(float(mesh_scale), 12))
        group = groups.setdefault(
            key,
            {"mesh_path": mesh_path, "mesh_scale": float(mesh_scale), "object_names": []},
        )
        group["object_names"].append(name)
    for group in groups.values():
        group["object_names"].sort()
        group["primary_name"] = group["object_names"][0]
    return sorted(groups.values(), key=lambda item: item["primary_name"])


def _valid_bank(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        count = len(FeaturePoseBank.load_npz(path))
    except Exception:
        return None
    return count if count > 0 else None


def _atomic_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _record_base(group: dict) -> dict:
    return {
        "primary_name": group["primary_name"],
        "object_names": list(group["object_names"]),
        "mesh_path": str(group["mesh_path"]),
        "mesh_scale": float(group["mesh_scale"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = _selected_specs(args.object_names)
    groups = _geometry_groups(specs)
    sam6d_root = Path(args.sam6d_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    cam_poses = Path(args.cam_poses).expanduser().resolve()
    if not cam_poses.exists():
        raise FileNotFoundError(cam_poses)

    encoder = None
    records = []
    started = time.time()
    for group_index, group in enumerate(groups, start=1):
        primary = group["primary_name"]
        bank_paths = {
            name: output_root / f"{name}_{args.dino_model_name}.npz" for name in group["object_names"]
        }
        existing = {name: _valid_bank(path) for name, path in bank_paths.items()}
        if not args.force_banks and all(count is not None for count in existing.values()):
            count = min(int(value) for value in existing.values() if value is not None)
            print(f"[rrtrack-all-banks] {group_index}/{len(groups)} reused {primary}: entries={count}")
            records.append({**_record_base(group), "bank_entries": count, "bank_paths": {k: str(v) for k, v in bank_paths.items()}, "reused": True})
            continue

        print(
            f"[rrtrack-all-banks] {group_index}/{len(groups)} geometry={primary} "
            f"aliases={','.join(group['object_names'])}"
        )
        group_work = work_root / primary
        group_work.mkdir(parents=True, exist_ok=True)
        cad_path = export_cad_mm(group["mesh_path"], group["mesh_scale"], group_work / "cad_mm.ply").resolve()
        template_args = SimpleNamespace(
            templates_dir=None,
            template_cache_root=str(Path(args.template_cache_root).expanduser().resolve()),
        )
        templates_dir = prepare_templates(
            template_args,
            sam6d_root,
            group_work,
            cad_path,
            object_name=primary,
            mesh_file=str(group["mesh_path"]),
            mesh_scale=group["mesh_scale"],
        )
        if encoder is None:
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
            templates_dir,
            cam_poses,
            encoder,
            base_views=args.base_views,
            progress=lambda done, total, index, entries, label=primary: print(
                f"[rrtrack-all-banks] {label} {done}/{total} template={index} entries={entries}"
            ),
        )
        primary_path = bank.save_npz(bank_paths[primary])
        for name, path in bank_paths.items():
            if name != primary:
                shutil.copy2(primary_path, path)
        records.append(
            {
                **_record_base(group),
                "templates_dir": str(Path(templates_dir).resolve()),
                "bank_entries": len(bank),
                "bank_paths": {key: str(value) for key, value in bank_paths.items()},
                "reused": False,
            }
        )
        _atomic_manifest(
            output_root / "manifest.json",
            {
                "schema_version": 1,
                "status": "building",
                "dino_model_name": args.dino_model_name,
                "cam_poses": str(cam_poses),
                "geometry_group_count": len(groups),
                "object_count": len(specs),
                "groups": records,
            },
        )

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dino_model_name": args.dino_model_name,
        "cam_poses": str(cam_poses),
        "geometry_group_count": len(groups),
        "object_count": len(specs),
        "elapsed_s": time.time() - started,
        "groups": records,
    }
    _atomic_manifest(output_root / "manifest.json", manifest)
    print(
        f"[rrtrack-all-banks] complete objects={len(specs)} geometries={len(groups)} "
        f"manifest={output_root / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
