from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = APP_ROOT / "rm75_app"
ASSET_DIR = APP_ROOT / "assets"
RUNTIME_DIR = APP_ROOT / "runtime_data"

# Compatibility name for older in-package code.  This no longer points at the
# original pick_jiaobang folder; all runtime state now stays inside APP_ROOT.
PICK_DIR = APP_ROOT

CONFIG_DIR = ASSET_DIR / "curobo_rm75_config"
MESH_DIR = ASSET_DIR / "meshs"
TEST_SCENE_DIR = ASSET_DIR / "test_scenes"
ROBOT_MODEL_DIR = ASSET_DIR / "robot_models"
DEFAULT_RM75_URDF = ROBOT_MODEL_DIR / "RM75_gripper" / "RM75-B" / "urdf" / "RM75-B.urdf"
DEFAULT_CAMERA_EXTRINSIC = ASSET_DIR / "calibration" / "camera_extrinsic_opencv.npy"
DEFAULT_CUROBO_CFG = CONFIG_DIR / "rm75.yml"
DEFAULT_SCENE_FILE = TEST_SCENE_DIR / "current_table.json"


def require_local_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def ensure_runtime_dirs() -> None:
    for name in (
        "failure_renders",
        "logs",
        "llm_pick_place_runs",
        "planning_profile_logs",
        "sam6d_grasp_scene_runs",
        "sam6d_groundingdino_runs",
        "sam6d_template_cache",
        "curobo_torch_extensions",
        "web_debug_packs",
        "web_control_logs",
        "web_process_logs",
        "calibration_runs",
    ):
        (RUNTIME_DIR / name).mkdir(parents=True, exist_ok=True)
