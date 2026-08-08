from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rm75_app.paths import APP_ROOT, RUNTIME_DIR


DEFAULT_WRIST_SCRIPT = APP_ROOT / "rm75_app" / "perception" / "wrist_gripper_object_relation.py"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_matrix_json(path: Path, key: str, matrix: np.ndarray) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(key): np.asarray(matrix, dtype=np.float64).reshape(4, 4)}
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected relation json object, got {type(payload).__name__}: {path}")
    return payload


def _matrix_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 16:
        return None
    return arr.reshape(4, 4)


_RUN_REPORT_NAMES = (
    "calibration_report.json",
    "wrist_anchor_optimization_report.json",
    "joint_optimization_report.json",
    "visual_refine_report.json",
    "combined_check_report.json",
)


def _calibration_roots() -> list[Path]:
    roots = [
        RUNTIME_DIR / "calibration_runs",
        APP_ROOT.parent / "runtime_data" / "calibration_runs",
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(root)
    return out


def _load_transform_file(path: Path) -> np.ndarray | None:
    path = Path(path).expanduser()
    try:
        if path.suffix == ".npy":
            return _matrix_or_none(np.load(path))
        payload = _load_json(path)
        if "matrix" in payload:
            return _matrix_or_none(payload.get("matrix"))
        for key in ("T_E_Cw", "ee_T_wrist_camera", "ee_T_wrist_camera_refined", "ee_T_wrist_camera_joint"):
            if key in payload:
                return _matrix_or_none(payload.get(key))
    except Exception:
        return None
    return None


def _is_identity_transform_file(path: Path, *, atol: float = 1e-9) -> bool:
    matrix = _load_transform_file(path)
    return bool(matrix is not None and np.allclose(matrix, np.eye(4, dtype=np.float64), atol=float(atol)))


def _run_acceptance(run_dir: Path) -> tuple[bool, bool]:
    """Return (accepted, has_explicit_accepted_field) for a calibration run."""

    run_dir = Path(run_dir).expanduser()
    status_seen: bool | None = None
    for name in _RUN_REPORT_NAMES:
        path = run_dir / name
        if not path.exists():
            continue
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if "accepted" in payload:
            return bool(payload.get("accepted")), True
        status = str(payload.get("status", "")).lower()
        if status in {"pass", "passed", "ok", "success"}:
            status_seen = True
        elif status in {"fail", "failed", "rejected", "error"}:
            status_seen = False
    if status_seen is not None:
        return bool(status_seen), False
    return True, False


def _accepted_run(run_dir: Path, *, require_explicit: bool) -> bool:
    accepted, explicit = _run_acceptance(run_dir)
    if require_explicit:
        return bool(accepted and explicit)
    return bool(accepted)


def _accepted_refined_copy(path: Path) -> bool:
    metadata_path = Path(path).expanduser().parent / "ee_T_wrist_camera_refined_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = _load_json(metadata_path)
    except Exception:
        return False
    return bool(metadata.get("accepted", False))


def _newest_file(paths: list[Path]) -> Path | None:
    existing = [Path(path).expanduser() for path in paths if Path(path).expanduser().exists()]
    existing = [path for path in existing if not _is_identity_transform_file(path)]
    if not existing:
        return None
    return max(existing, key=lambda path: (path.stat().st_mtime, 1 if path.suffix == ".npy" else 0))


def _candidate_files(run_dir: Path, names: tuple[str, ...]) -> list[Path]:
    return [Path(run_dir).expanduser() / name for name in names]


def latest_hand_eye_path() -> Path | None:
    """Return the best automatically discovered wrist hand-eye transform.

    Priority is deliberately stricter than mtime-only lookup: accepted anchored
    and joint-board results win, accepted visual refinements copied into a
    trusted board run are next, and classical wrist-board calibration is the
    fallback. Rejected or identity smoke-test outputs are ignored.
    """

    roots = _calibration_roots()

    for pattern, filenames in (
        ("*_wrist_camera_dual_board", ("ee_T_wrist_camera.npy", "ee_T_wrist_camera.json")),
        ("*_wrist_camera_board_anchor", ("ee_T_wrist_camera.npy", "ee_T_wrist_camera.json")),
        ("*_two_camera_joint_board", ("ee_T_wrist_camera_joint.npy", "ee_T_wrist_camera_joint.json")),
    ):
        candidates: list[Path] = []
        for root in roots:
            for run_dir in root.glob(pattern):
                if not run_dir.is_dir() or not _accepted_run(run_dir, require_explicit=True):
                    continue
                path = _newest_file(_candidate_files(run_dir, filenames))
                if path is not None:
                    candidates.append(path)
        if candidates:
            return max(candidates, key=lambda path: (path.stat().st_mtime, 1 if path.suffix == ".npy" else 0))

    refined_candidates: list[Path] = []
    for root in roots:
        for run_dir in root.glob("*_wrist_camera_board"):
            if not run_dir.is_dir() or not _accepted_run(run_dir, require_explicit=False):
                continue
            path = _newest_file(
                _candidate_files(run_dir, ("ee_T_wrist_camera_refined.npy", "ee_T_wrist_camera_refined.json"))
            )
            if path is not None and _accepted_refined_copy(path):
                refined_candidates.append(path)
    if refined_candidates:
        return max(refined_candidates, key=lambda path: (path.stat().st_mtime, 1 if path.suffix == ".npy" else 0))

    board_candidates: list[Path] = []
    for root in roots:
        for run_dir in root.glob("*_wrist_camera_board"):
            if not run_dir.is_dir() or not _accepted_run(run_dir, require_explicit=False):
                continue
            path = _newest_file(_candidate_files(run_dir, ("ee_T_wrist_camera.npy", "ee_T_wrist_camera.json")))
            if path is not None:
                board_candidates.append(path)
    if board_candidates:
        return max(board_candidates, key=lambda path: (path.stat().st_mtime, 1 if path.suffix == ".npy" else 0))
    return None


@dataclass
class WristRelationConfig:
    object_name: str
    script_path: Path = DEFAULT_WRIST_SCRIPT
    python: str = sys.executable
    output_dir: Path = RUNTIME_DIR / "wrist_relation_runs"
    relation_out: Path = RUNTIME_DIR / "wrist_relation_latest.json"
    hand_eye_path: Path | None = None
    init_T_gripper_obj_path: Path | None = None
    realsense_serial: str = ""
    prompt: str | None = None
    timeout_s: float = 45.0
    min_confidence: float = 0.25
    min_projection_iou: float = 0.08
    require_held: bool = True
    extra_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class WristRelationEstimate:
    ok: bool
    accepted: bool
    object_name: str
    relation_path: Path
    T_gripper_obj: np.ndarray | None
    confidence: float
    projection_iou: float
    grasp_state: str
    reason: str
    payload: dict[str, Any]
    returncode: int | None = None


class WristRelationJob:
    def __init__(self, config: WristRelationConfig):
        self.config = config
        self._done = threading.Event()
        self._result: WristRelationEstimate | None = None
        self._error: BaseException | None = None
        self._start_time = time.perf_counter()
        self._done_time: float | None = None
        self._thread = threading.Thread(target=self._run, name="wrist-relation", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._result = run_once(self.config)
        except BaseException as exc:
            self._error = exc
        finally:
            self._done_time = time.perf_counter()
            self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout_s: float | None = None) -> WristRelationEstimate | None:
        if not self._done.wait(timeout=None if timeout_s is None else max(0.0, float(timeout_s))):
            return None
        if self._error is not None:
            raise self._error
        return self._result

    def elapsed_s(self) -> float | None:
        if self._done_time is None:
            return None
        return float(self._done_time - self._start_time)


def build_command(config: WristRelationConfig) -> list[str]:
    script = Path(config.script_path).expanduser()
    hand_eye = Path(config.hand_eye_path).expanduser() if config.hand_eye_path else latest_hand_eye_path()
    if hand_eye is None:
        raise FileNotFoundError(
            "No wrist hand-eye transform found. Pass --wrist-relation-hand-eye-path "
            "or run calib-wrist / calib-wrist-anchor first to create an accepted ee_T_wrist_camera.npy/json, "
            "or calib-joint-board to create an accepted ee_T_wrist_camera_joint.npy/json."
        )
    cmd = [
        str(config.python),
        str(script),
        "--object-name",
        str(config.object_name),
        "--output-dir",
        str(Path(config.output_dir).expanduser()),
        "--relation-out",
        str(Path(config.relation_out).expanduser()),
        "--T-gripper-cam-path",
        str(hand_eye),
        "--min-confidence",
        str(float(config.min_confidence)),
        "--min-projection-iou",
        str(float(config.min_projection_iou)),
    ]
    if config.prompt:
        cmd.extend(["--prompt", str(config.prompt)])
    if config.realsense_serial:
        cmd.extend(["--realsense-serial", str(config.realsense_serial)])
    if config.init_T_gripper_obj_path is not None:
        cmd.extend(["--init-T-gripper-obj-path", str(Path(config.init_T_gripper_obj_path).expanduser())])
    cmd.extend(str(arg) for arg in config.extra_args)
    return cmd


def parse_relation(path: Path, *, require_held: bool = True, returncode: int | None = None) -> WristRelationEstimate:
    path = Path(path).expanduser()
    payload = _load_json(path)
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    projection_iou = float(payload.get("projection_iou", 0.0) or 0.0)
    T_gripper_obj = _matrix_or_none(payload.get("T_gripper_obj"))
    grasp_payload = payload.get("grasp_state") if isinstance(payload.get("grasp_state"), dict) else {}
    grasp_state = str(grasp_payload.get("state", "unknown") if isinstance(grasp_payload, dict) else "unknown")
    accepted = bool(payload.get("ok", False)) and T_gripper_obj is not None
    if require_held and grasp_state not in {"held", "likely_held"}:
        accepted = False
    reason = str(payload.get("reason", "accepted" if accepted else "rejected"))
    if require_held and grasp_state not in {"held", "likely_held"}:
        reason = f"grasp_state_{grasp_state}"
    return WristRelationEstimate(
        ok=bool(payload.get("ok", False)),
        accepted=bool(accepted),
        object_name=str(payload.get("object_name", "")),
        relation_path=path,
        T_gripper_obj=T_gripper_obj,
        confidence=confidence,
        projection_iou=projection_iou,
        grasp_state=grasp_state,
        reason=reason,
        payload=payload,
        returncode=returncode,
    )


def run_once(config: WristRelationConfig) -> WristRelationEstimate:
    cmd = build_command(config)
    script_path = Path(config.script_path).expanduser()
    if not script_path.exists():
        raise FileNotFoundError(f"wrist relation script not found: {script_path}")
    relation_out = Path(config.relation_out).expanduser()
    relation_out.parent.mkdir(parents=True, exist_ok=True)
    if relation_out.exists():
        relation_out.unlink()
    t0 = time.perf_counter()
    print("[wrist_relation] running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(APP_ROOT.parent), text=True, timeout=float(config.timeout_s))
    elapsed = time.perf_counter() - t0
    if not relation_out.exists():
        raise FileNotFoundError(f"wrist relation did not write relation json: {relation_out}")
    result = parse_relation(relation_out, require_held=bool(config.require_held), returncode=int(proc.returncode))
    print(
        f"[wrist_relation] done rc={proc.returncode} elapsed_s={elapsed:.2f} "
        f"accepted={result.accepted} conf={result.confidence:.3f} "
        f"iou={result.projection_iou:.3f} grasp={result.grasp_state}"
    )
    if proc.returncode != 0:
        result.accepted = False
        result.reason = f"subprocess_rc_{proc.returncode}"
    return result


def start_async(config: WristRelationConfig) -> WristRelationJob:
    return WristRelationJob(config)
