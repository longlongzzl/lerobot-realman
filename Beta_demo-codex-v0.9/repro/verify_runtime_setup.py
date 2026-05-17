from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gymnasium as gym


PATCH_FILES = [
    Path("mani_skill/agents/robots/realman/__init__.py"),
    Path("mani_skill/agents/robots/realman/realman_with_gripper.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", type=Path, default=None)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def patch_root() -> Path:
    return Path(__file__).resolve().parent / "maniskill_patch"


def curobo_snapshot_src() -> Path:
    return Path(__file__).resolve().parent / "curobo_snapshot" / "src"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def bootstrap_curobo_snapshot() -> Path:
    src = curobo_snapshot_src().resolve()
    if not (src / "curobo").is_dir():
        raise FileNotFoundError(f"Missing CuRobo snapshot: {src}")
    sys.path[:] = [p for p in sys.path if "curobo-v078" not in p.lower()]
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.7.8.post1.dev0+dirty")
    os.environ.setdefault("VCS_VERSIONING_PRETEND_VERSION", "0.7.8.post1.dev0+dirty")
    return src


def main() -> int:
    args = parse_args()
    repo_root_str = str(repo_root())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    curobo_src = bootstrap_curobo_snapshot()

    from curobo.geom.types import WorldConfig
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig

    print(f"[verify-runtime] curobo_snapshot={curobo_src}")
    print(f"[verify-runtime] curobo_pose={Pose}")
    print(f"[verify-runtime] curobo_robot_config={RobotConfig}")
    print(f"[verify-runtime] curobo_world_config={WorldConfig}")

    import mani_skill
    from mani_skill import ASSET_DIR

    runtime_root = Path(inspect.getfile(mani_skill)).resolve().parent.parent
    target_root = normalize_target_root((args.target_root or detect_editable_root() or runtime_root).resolve())

    print(f"[verify-runtime] mani_skill_init={inspect.getfile(mani_skill)}")
    print(f"[verify-runtime] target_root={target_root}")

    for rel in PATCH_FILES:
        repo_file = patch_root() / rel
        runtime_file = target_root / rel
        if not runtime_file.is_file():
            raise FileNotFoundError(f"Missing runtime file: {runtime_file}")
        repo_hash = sha256(repo_file)
        runtime_hash = sha256(runtime_file)
        if repo_hash != runtime_hash:
            raise RuntimeError(f"Hash mismatch for {rel}: repo={repo_hash} runtime={runtime_hash}")
        print(f"[verify-runtime] ok={rel.as_posix()} sha256={repo_hash}")

    rm75_urdf = Path(ASSET_DIR) / "robots" / "RM75_gripper" / "RM75-B" / "urdf" / "RM75-B.urdf"
    if not rm75_urdf.is_file():
        raise FileNotFoundError(f"Missing RM75 URDF asset: {rm75_urdf}")
    print(f"[verify-runtime] rm75_urdf={rm75_urdf}")

    import jimu_pick_cube_env  # noqa: F401

    spec = gym.spec("JimuPickCube-v1")
    print(f"[verify-runtime] env_id={spec.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
