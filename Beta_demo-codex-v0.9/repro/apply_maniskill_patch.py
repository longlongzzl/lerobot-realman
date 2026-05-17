from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", type=Path, default=None)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def patch_root() -> Path:
    return Path(__file__).resolve().parent / "maniskill_patch"


def detect_editable_root() -> Path | None:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "show", "mani_skill"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        if line.startswith("Editable project location:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return Path(value)
    return None


def normalize_target_root(path: Path) -> Path:
    if (path / "mani_skill").is_dir():
        return path
    if path.name == "mani_skill" and path.is_dir():
        return path.parent
    raise FileNotFoundError(f"Cannot locate mani_skill package root under: {path}")


def iter_patch_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def main() -> int:
    args = parse_args()
    source_root = patch_root()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Patch root not found: {source_root}")

    target_root = args.target_root or detect_editable_root()
    if target_root is None:
        raise RuntimeError("Could not detect editable mani_skill root. Pass --target-root explicitly.")
    target_root = normalize_target_root(target_root.resolve())

    copied: list[str] = []
    for src in iter_patch_files(source_root):
        rel = src.relative_to(source_root)
        dst = target_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(rel).replace("\\", "/"))

    print(f"[apply-maniskill-patch] repo_root={repo_root()}")
    print(f"[apply-maniskill-patch] target_root={target_root}")
    for rel in copied:
        print(f"[apply-maniskill-patch] copied={rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
