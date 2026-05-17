from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .profile_cache import update_profile_cache
from .retry_policy import classify_failure as decide_failure
from .schemas import AttemptResult, PerturbationCase, VideoManifestItem


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("BETA_DEMO_PYTHON", sys.executable))
PLANNER = ROOT / "plan_second_layer_triangle_wall_path.py"
CHECKPOINT = ROOT / "v6_standard_four_wall_retry_build_v1" / "checkpoints" / "stable_04_front_wall_state.json"

DEFAULT_CASES = [
    PerturbationCase("p01_dx+020_dy+000_yaw+5", 0.020, 0.000, 5.0),
    PerturbationCase("p02_dx-020_dy+000_yaw-5", -0.020, 0.000, -5.0),
    PerturbationCase("p03_dx+030_dy-020_yaw+8", 0.030, -0.020, 8.0),
    PerturbationCase("p04_dx-030_dy+020_yaw-8", -0.030, 0.020, -8.0),
    PerturbationCase("p05_dx+040_dy-025_yaw+10", 0.040, -0.025, 10.0),
    PerturbationCase("p06_dx-040_dy+025_yaw-10", -0.040, 0.025, -10.0),
    PerturbationCase("p07_dx+050_dy+030_yaw+12", 0.050, 0.030, 12.0),
    PerturbationCase("p08_dx-050_dy-030_yaw-12", -0.050, -0.030, -12.0),
]


@dataclass
class MatrixConfig:
    out_root: Path = ROOT / "v8_modular_robustness_matrix"
    start_index: int = 0
    limit: int = 0
    case_retries: int = 2
    parallel_workers: int = 1
    resume: bool = False
    release_gap_mms: str = "3"
    release_yaw_degs: str = "0,-1,1"
    max_release_candidates: int = 36
    release_ik_max_position_error: str = "0.009"
    release_ik_max_rotation_error: str = "0.18"
    edge_seating_attempts: int = 0
    edge_seating_max_step: str = "0.0"
    edge_seating_max_angle_step_deg: str = "2.0"
    pre_magnet_geometric_capture_attempts: int = 0
    magnetic_capture_nudge_attempts: int = 0
    magnetic_capture_nudge_step: str = "0.002"
    magnetic_capture_nudge_steps: str = "10"
    magnetic_capture_hold_steps: str = "10"
    magnetic_capture_max_angle_step_deg: str = "2.0"
    magnetic_capture_max_joint_delta: str = "1.4"
    magnetic_capture_revert_tolerance: str = "0.0007"

    @classmethod
    def from_namespace(cls, args: Any) -> MatrixConfig:
        return cls(
            out_root=Path(args.out_root),
            start_index=int(args.start_index),
            limit=int(args.limit),
            case_retries=int(args.case_retries),
            parallel_workers=max(int(getattr(args, "parallel_workers", 1)), 1),
            resume=bool(args.resume),
            release_gap_mms=str(args.release_gap_mms),
            release_yaw_degs=str(args.release_yaw_degs),
            max_release_candidates=int(args.max_release_candidates),
            release_ik_max_position_error=str(args.release_ik_max_position_error),
            release_ik_max_rotation_error=str(args.release_ik_max_rotation_error),
            edge_seating_attempts=int(args.edge_seating_attempts),
            edge_seating_max_step=str(args.edge_seating_max_step),
            edge_seating_max_angle_step_deg=str(args.edge_seating_max_angle_step_deg),
            pre_magnet_geometric_capture_attempts=int(args.pre_magnet_geometric_capture_attempts),
            magnetic_capture_nudge_attempts=int(args.magnetic_capture_nudge_attempts),
            magnetic_capture_nudge_step=str(args.magnetic_capture_nudge_step),
            magnetic_capture_nudge_steps=str(args.magnetic_capture_nudge_steps),
            magnetic_capture_hold_steps=str(args.magnetic_capture_hold_steps),
            magnetic_capture_max_angle_step_deg=str(args.magnetic_capture_max_angle_step_deg),
            magnetic_capture_max_joint_delta=str(args.magnetic_capture_max_joint_delta),
            magnetic_capture_revert_tolerance=str(args.magnetic_capture_revert_tolerance),
        )


def physical_env() -> dict[str, str]:
    env = dict(os.environ)
    env["JIMU_DRIVE_FORCE_LIMIT"] = "8.0"
    env["JIMU_DRIVE_STIFFNESS"] = "85.0"
    env["JIMU_DRIVE_DAMPING"] = "14.0"
    env["JIMU_DRIVE_ANGULAR_STIFFNESS"] = "0.42"
    env["JIMU_DRIVE_ANGULAR_DAMPING"] = "0.060"
    env["JIMU_DRIVE_ANGULAR_FORCE_LIMIT"] = "0.55"
    env["JIMU_ATTRACT_STIFFNESS"] = "6.0"
    env["JIMU_ATTRACT_FORCE_LIMIT"] = "2.5"
    env["JIMU_ATTRACT_TORQUE_STIFFNESS"] = "0.20"
    env["JIMU_ATTRACT_TORQUE_LIMIT"] = "0.14"
    env["JIMU_ATTRACT_NORMAL_TORQUE_STIFFNESS"] = "0.48"
    env["JIMU_ATTRACT_NORMAL_TORQUE_LIMIT"] = "0.32"
    return env


def select_cases(config: MatrixConfig) -> list[tuple[int, PerturbationCase]]:
    selected = DEFAULT_CASES[int(config.start_index) :]
    if int(config.limit) > 0:
        selected = selected[: int(config.limit)]
    return [(int(config.start_index) + offset, case) for offset, case in enumerate(selected)]


def build_second_layer_command(out_dir: Path, case: PerturbationCase, config: MatrixConfig) -> list[str]:
    return [
        str(PYTHON),
        str(PLANNER),
        "--out-dir",
        str(out_dir),
        "--load-assembly-state",
        str(CHECKPOINT),
        "--loaded-state-perturb-dx",
        f"{case.dx:.6f}",
        "--loaded-state-perturb-dy",
        f"{case.dy:.6f}",
        "--loaded-state-perturb-yaw-deg",
        f"{case.yaw:.6f}",
        "--roles",
        "front_second_triangle,right_second_triangle,left_second_triangle,back_second_triangle",
        "--record-live",
        "--fps",
        "12",
        "--record-every",
        "4",
        "--add-triangle-staging-fixtures",
        "--magnet-attach-distance",
        "0.010",
        "--magnet-attract-distance",
        "0.024",
        "--magnet-detach-distance",
        "0.035",
        "--max-grasp-candidates",
        "8",
        "--ik-seeds",
        "64",
        "--move-steps",
        "40",
        "--release-steps",
        "40",
        "--max-joint-step",
        "0.024",
        "--max-segment-steps",
        "1500",
        "--preplace-heights",
        "0.026,0.034",
        "--release-gap-mms",
        str(config.release_gap_mms),
        f"--release-yaw-degs={config.release_yaw_degs}",
        "--max-release-candidates",
        str(config.max_release_candidates),
        "--release-ik-max-position-error",
        str(config.release_ik_max_position_error),
        "--release-ik-max-rotation-error",
        str(config.release_ik_max_rotation_error),
        "--preplace-ik-max-position-error",
        "0.012",
        "--preplace-ik-max-rotation-error",
        "0.22",
        "--min-parent-top-edge-z",
        "0.045",
        "--release-preplace-max-joint-delta",
        "5.3",
        "--release-max-joint-delta",
        "2.4",
        "--use-screened-release-after-lift",
        "--no-allow-screened-release-fallback",
        "--require-active-connection-before-open",
        "--pre-magnet-geometric-capture-attempts",
        str(config.pre_magnet_geometric_capture_attempts),
        "--magnetic-capture-nudge-attempts",
        str(config.magnetic_capture_nudge_attempts),
        "--magnetic-capture-nudge-step",
        str(config.magnetic_capture_nudge_step),
        "--magnetic-capture-nudge-steps",
        str(config.magnetic_capture_nudge_steps),
        "--magnetic-capture-hold-steps",
        str(config.magnetic_capture_hold_steps),
        "--magnetic-capture-max-angle-step-deg",
        str(config.magnetic_capture_max_angle_step_deg),
        "--magnetic-capture-max-joint-delta",
        str(config.magnetic_capture_max_joint_delta),
        "--magnetic-capture-revert-if-worse",
        "--magnetic-capture-revert-tolerance",
        str(config.magnetic_capture_revert_tolerance),
        "--pre-open-hold-steps",
        "0",
        "--open-steps",
        "36",
        "--post-open-hold-steps",
        "72",
        "--release-correction-attempts",
        "0",
        "--release-correction-edge-threshold",
        "0.010",
        "--release-correction-revert-if-worse",
        "--edge-seating-attempts",
        str(config.edge_seating_attempts),
        "--edge-seating-max-step",
        str(config.edge_seating_max_step),
        "--edge-seating-max-angle-step-deg",
        str(config.edge_seating_max_angle_step_deg),
        "--edge-seating-rotation-mode",
        "world_z",
        "--edge-seating-max-error",
        "0.010",
        "--edge-seating-center-error",
        "0.010",
        "--edge-seating-angle-error-deg",
        "10",
        "--stability-steps",
        "140",
        "--final-all-roles-stability-steps",
        "50",
        "--final-max-base-edge-error",
        "0.010",
        "--final-max-base-center-error",
        "0.0095",
        "--final-max-base-edge-angle-deg",
        "10",
        "--pre-open-max-base-edge-error",
        "0.024",
        "--pre-open-max-base-center-error",
        "0.020",
        "--pre-open-max-base-edge-angle-deg",
        "22",
        "--release-open-gripper-value",
        "0.03",
        "--post-success-safe-lift-steps",
        "48",
        "--post-success-safe-lift-height",
        "0.06",
        "--retreat-mode",
        "tcp_back",
        "--retreat-distance",
        "0.025",
        "--return-home-steps",
        "56",
        "--anchor-floor-during-build",
        "--no-release-floor-anchor-before-final",
    ]


def classify_failure(failed_at: str, returncode: int) -> tuple[str, bool]:
    decision = decide_failure(failed_at, returncode)
    return decision.failure_class, decision.retryable


def extract_case_result(out_dir: Path, case: PerturbationCase, returncode: int) -> AttemptResult:
    summary_path = out_dir / "triangle_wall_summary.json"
    video_path = out_dir / "triangle_wall_live.mp4"
    result = AttemptResult(
        case=case.to_json(),
        out_dir=str(out_dir),
        returncode=int(returncode),
        summary=str(summary_path) if summary_path.exists() else "",
        video=str(video_path) if video_path.exists() else "",
        success=False,
        failed_at="missing_summary" if not summary_path.exists() else "",
    )
    if not summary_path.exists():
        result.failure_class, result.retryable = classify_failure(result.failed_at, returncode)
        return result
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result.failed_at = f"summary_read_error:{type(exc).__name__}:{exc}"
        result.failure_class, result.retryable = classify_failure(result.failed_at, returncode)
        return result
    final = data.get("final", {})
    reports = data.get("reports", [])
    result.success = bool(final.get("success", False))
    result.completed_roles = list(final.get("completed_roles", []))
    if not result.success:
        failed = [f"{item.get('role')}:{item.get('failed_at')}" for item in reports if not item.get("success")]
        result.failed_at = ", ".join(failed) or "final_failed"
    edge = final.get("final_triangle_edge_alignment", {})
    result.edge_errors_m = {
        role: float(info.get("max_point_error_m", float("nan")))
        for role, info in edge.items()
        if isinstance(info, dict) and info.get("success")
    }
    result.edge_angles_deg = {
        role: float(info.get("edge_parallel_error_deg", float("nan")))
        for role, info in edge.items()
        if isinstance(info, dict) and info.get("success")
    }
    conn = final.get("final_triangle_connection_point_error", {})
    result.attach_distances_m = {
        role: float(info.get("attach_distance_m", float("nan")))
        for role, info in conn.items()
        if isinstance(info, dict)
    }
    result.failure_class, result.retryable = classify_failure(result.failed_at, returncode)
    return result


def run_case(index: int, case: PerturbationCase, config: MatrixConfig, env: dict[str, str]) -> dict[str, Any]:
    out_root = Path(config.out_root)
    case_name = case.name
    base_out_dir = out_root / f"case_{index + 1:02d}_{case_name}"
    case_log_path = base_out_dir / "matrix_case.log"
    summary_path = base_out_dir / "triangle_wall_summary.json"
    if config.resume and summary_path.exists():
        result = extract_case_result(base_out_dir, case, 0)
        payload = result.to_json()
        payload["attempts"] = [result.to_json()]
        return payload

    attempts: list[dict[str, Any]] = []
    max_attempts = max(int(config.case_retries), 0) + 1
    last_result: AttemptResult | None = None
    for attempt_index in range(max_attempts):
        out_dir = base_out_dir if attempt_index == 0 else out_root / f"case_{index + 1:02d}_{case_name}_retry{attempt_index + 1:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if attempt_index == 0:
            base_out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_second_layer_command(out_dir, case, config)
        (out_dir / "command.json").write_text(json.dumps(cmd, ensure_ascii=False, indent=2), encoding="utf-8")
        with case_log_path.open("a", encoding="utf-8") as case_log:
            case_log.write(
                f"[run] {datetime.now().isoformat(timespec='seconds')} {case_name} "
                f"attempt={attempt_index + 1}/{max_attempts} out={out_dir}\n"
            )
        stdout_path = out_dir / "run_stdout.log"
        stderr_path = out_dir / "run_stderr.log"
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr_file:
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                check=False,
            )
        last_result = extract_case_result(out_dir, case, int(completed.returncode))
        last_result.attempt = attempt_index + 1
        attempts.append(last_result.to_json())
        with case_log_path.open("a", encoding="utf-8") as case_log:
            case_log.write(
                f"[done] {datetime.now().isoformat(timespec='seconds')} {case_name} "
                f"attempt={attempt_index + 1} returncode={completed.returncode} "
                f"success={last_result.success} failure_class={last_result.failure_class} "
                f"failed_at={last_result.failed_at}\n"
            )
        if last_result.success or not last_result.retryable:
            break
    assert last_result is not None
    payload = last_result.to_json()
    payload["attempts"] = attempts
    return payload


def write_video_manifest(out_root: Path, results: list[dict[str, Any]]) -> None:
    items: list[VideoManifestItem] = []
    for result in results:
        for attempt in result.get("attempts", []):
            video = str(attempt.get("video", "") or "")
            if not video:
                continue
            items.append(
                VideoManifestItem(
                    case_name=str((attempt.get("case") or {}).get("name", "")),
                    attempt=int(attempt.get("attempt", 1)),
                    success=bool(attempt.get("success", False)),
                    failure_class=str(attempt.get("failure_class", "")),
                    failed_at=str(attempt.get("failed_at", "")),
                    video=Path(video),
                )
            )
    (out_root / "video_manifest.json").write_text(
        json.dumps([item.to_json() for item in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = ["# Jimu planner video manifest", ""]
    for item in items:
        status = "success" if item.success else f"failed:{item.failure_class}"
        video_text = str(item.video).replace("\\", "/")
        lines.append(f"## {item.case_name} attempt {item.attempt} {status}")
        if item.failed_at:
            lines.append(f"- failed_at: `{item.failed_at}`")
        lines.append(f"![{item.case_name} attempt {item.attempt}]({video_text})")
        lines.append("")
    (out_root / "video_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def matrix_payload(out_root: Path, results: list[dict[str, Any]], config: MatrixConfig) -> dict[str, Any]:
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(out_root),
        "total": len(DEFAULT_CASES),
        "finished": len(results),
        "success_count": sum(1 for item in results if item.get("success")),
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()
        },
        "results": results,
    }


def update_profiles_from_results(out_root: Path, results: list[dict[str, Any]]) -> None:
    summary_paths: list[Path] = []
    for result in results:
        for attempt in result.get("attempts", []):
            if not attempt.get("success"):
                continue
            summary = str(attempt.get("summary", "") or "")
            if summary:
                summary_paths.append(Path(summary))
    if summary_paths:
        update_profile_cache(out_root / "profile_cache.json", summary_paths)


def run_matrix(config: MatrixConfig) -> dict[str, Any]:
    out_root = Path(config.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "matrix_results.json"
    selected = select_cases(config)
    env = physical_env()
    results_by_index: dict[int, dict[str, Any]] = {}
    if config.resume and results_path.exists():
        try:
            existing = json.loads(results_path.read_text(encoding="utf-8"))
            for item in existing.get("results", []):
                if not item.get("attempts"):
                    continue
                name = str((item.get("case") or {}).get("name", ""))
                for case_index, case in selected:
                    if case.name == name:
                        results_by_index[case_index] = item
        except Exception:
            results_by_index = {}

    if int(config.parallel_workers) <= 1:
        for case_index, case in selected:
            if case_index in results_by_index and config.resume:
                continue
            results_by_index[case_index] = run_case(case_index, case, config, env)
            ordered = [results_by_index[index] for index, _case in selected if index in results_by_index]
            results_path.write_text(json.dumps(matrix_payload(out_root, ordered, config), ensure_ascii=False, indent=2), encoding="utf-8")
            write_video_manifest(out_root, ordered)
            update_profiles_from_results(out_root, ordered)
    else:
        with ThreadPoolExecutor(max_workers=int(config.parallel_workers)) as executor:
            futures = {
                executor.submit(run_case, case_index, case, config, env): case_index
                for case_index, case in selected
                if not (case_index in results_by_index and config.resume)
            }
            for future in as_completed(futures):
                case_index = futures[future]
                results_by_index[case_index] = future.result()
                ordered = [results_by_index[index] for index, _case in selected if index in results_by_index]
                results_path.write_text(json.dumps(matrix_payload(out_root, ordered, config), ensure_ascii=False, indent=2), encoding="utf-8")
                write_video_manifest(out_root, ordered)
                update_profiles_from_results(out_root, ordered)

    ordered = [results_by_index[index] for index, _case in selected if index in results_by_index]
    payload = matrix_payload(out_root, ordered, config)
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_video_manifest(out_root, ordered)
    update_profiles_from_results(out_root, ordered)
    return payload
