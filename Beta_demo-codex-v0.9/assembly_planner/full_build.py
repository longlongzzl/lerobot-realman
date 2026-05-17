from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .assembly_spec import standard_two_layer_wall_spec
from .profile_cache import update_profile_cache
from .schemas import PerturbationCase
from .second_layer_matrix import CHECKPOINT, PYTHON, ROOT, build_second_layer_command, physical_env


FOUR_WALL_SCRIPT = ROOT / "standard_four_wall_retry_build.py"


@dataclass
class FullBuildConfig:
    out_dir: Path = ROOT / "v9_full_physical_build"
    from_scratch: bool = False
    four_wall_state: Path = CHECKPOINT
    four_wall_max_attempts_per_role: int = 2
    four_wall_timeout_sec: float = 600.0
    triangle_case_retries: int = 2
    triangle_timeout_sec: float = 900.0
    enable_triangle_fallback: bool = True
    record_every: int = 4
    fps: int = 12
    use_success_profile_fast_path: bool = True


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def _run_command(command: list[str], out_dir: Path, timeout_sec: float | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    env = physical_env()
    try:
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr_file:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
                timeout=timeout_sec,
            )
        return {
            "returncode": int(completed.returncode),
            "timed_out": False,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": 124,
            "timed_out": True,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }


def four_wall_command(config: FullBuildConfig, out_dir: Path) -> list[str]:
    return [
        str(PYTHON),
        str(FOUR_WALL_SCRIPT),
        "--out-dir",
        str(out_dir),
        "--no-unique-out-dir",
        "--execution-mode",
        "direct-first",
        "--max-attempts-per-role",
        str(config.four_wall_max_attempts_per_role),
        "--attempt-timeout-sec",
        str(config.four_wall_timeout_sec),
        "--fps",
        str(config.fps),
        "--record-every",
        str(max(int(config.record_every), 1)),
        "--drive-force-limit",
        "5.0",
        "--magnet-attach-distance",
        "0.010",
        "--magnet-attract-distance",
        "0.024",
        "--magnet-detach-distance",
        "0.035",
        "--loaded-state-settle-steps",
        "8",
        "--release-open-gripper-value",
        "0.03",
        "--pre-open-connection-correction-attempts",
        "1",
        "--release-correction-attempts",
        "1",
        "--return-neutral-steps",
        "8",
    ]


def triangle_command(config: FullBuildConfig, out_dir: Path, state_path: Path, roles: str | None = None) -> list[str]:
    matrix_config = type(
        "TriangleConfig",
        (),
        {
            "release_gap_mms": "3",
            "release_yaw_degs": "0,-1,1",
            "max_release_candidates": 36,
            "release_ik_max_position_error": "0.009",
            "release_ik_max_rotation_error": "0.18",
            "pre_magnet_geometric_capture_attempts": 0,
            "magnetic_capture_nudge_attempts": 0,
            "magnetic_capture_nudge_step": "0.002",
            "magnetic_capture_nudge_steps": "10",
            "magnetic_capture_hold_steps": "10",
            "magnetic_capture_max_angle_step_deg": "2.0",
            "magnetic_capture_max_joint_delta": "1.4",
            "magnetic_capture_revert_tolerance": "0.0007",
            "edge_seating_attempts": 0,
            "edge_seating_max_step": "0.0",
            "edge_seating_max_angle_step_deg": "2.0",
        },
    )()
    case = PerturbationCase("nominal", 0.0, 0.0, 0.0)
    command = build_second_layer_command(out_dir, case, matrix_config)
    load_index = command.index("--load-assembly-state") + 1
    command[load_index] = str(state_path)
    roles_index = command.index("--roles") + 1
    if roles:
        command[roles_index] = str(roles)
    command.extend(["--save-assembly-state", str(out_dir / "triangle_assembly_state.json")])
    fps_index = command.index("--fps") + 1
    command[fps_index] = str(config.fps)
    record_index = command.index("--record-every") + 1
    command[record_index] = str(max(int(config.record_every), 1))
    if config.use_success_profile_fast_path:
        grasp_index = command.index("--max-grasp-candidates") + 1
        command[grasp_index] = "1"
        release_candidates_index = command.index("--max-release-candidates") + 1
        command[release_candidates_index] = "36"
        command.extend(
            [
                "--release-candidate-index-groups",
                (
                    "front_second_triangle:10;11;12;13;14,"
                    "right_second_triangle:0;1;2;3;4;5;6;7;8,"
                    "left_second_triangle:24;25;26;27;28;29;30,"
                    "back_second_triangle:0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25;26;27;28;29;30;31;32;33;34;35"
                ),
                "--preplace-heights-by-role",
                "front_second_triangle:0.055,right_second_triangle:0.034,left_second_triangle:0.034,back_second_triangle:0.026;0.034;0.055",
                "--preplace-height-by-role",
                "front_second_triangle:0.055,right_second_triangle:0.034,left_second_triangle:0.034,back_second_triangle:0.034",
                "--final-max-base-edge-error",
                "0.010",
                "--final-max-base-center-error",
                "0.008",
                "--final-max-base-edge-angle-deg",
                "8.0",
            ]
        )
    return command


def _remaining_triangle_roles(final: dict[str, Any]) -> str:
    roles = [str(item) for item in final.get("roles", []) if str(item)]
    completed = {str(item) for item in final.get("completed_roles", []) if str(item)}
    remaining = [role for role in roles if role not in completed]
    return ",".join(remaining)


def _triangle_role_order(attempt_index: int) -> str:
    orders = [
        "front_second_triangle,right_second_triangle,left_second_triangle,back_second_triangle",
        "front_second_triangle,back_second_triangle,left_second_triangle,right_second_triangle",
        "front_second_triangle,left_second_triangle,back_second_triangle,right_second_triangle",
        "front_second_triangle,back_second_triangle,right_second_triangle,left_second_triangle",
    ]
    return orders[int(attempt_index) % len(orders)]


def _video_items_from_four_wall(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not report:
        return []
    items: list[dict[str, Any]] = []
    for item in report.get("successes", []):
        video = str(item.get("video", "") or "")
        if video:
            items.append(
                {
                    "stage": "four_wall",
                    "role": item.get("role"),
                    "attempt_number": item.get("attempt_number"),
                    "success": True,
                    "video": video,
                }
            )
    return items


def write_full_video_manifest(out_dir: Path, items: list[dict[str, Any]]) -> None:
    (out_dir / "video_manifest.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Jimu full build video manifest", ""]
    for item in items:
        label = f"{item.get('stage')} {item.get('role', '')}".strip()
        video = str(item.get("video", "")).replace("\\", "/")
        lines.append(f"## {label}")
        lines.append(f"![{label}]({video})")
        lines.append("")
    (out_dir / "video_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def run_full_build(config: FullBuildConfig) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    standard_two_layer_wall_spec().save(out_dir / "assembly_spec.json")
    report: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()
        },
        "physical_boundary": {
            "magnet_attach_distance": "0.010",
            "magnet_attract_distance": "0.024",
            "magnet_detach_distance": "0.035",
        },
        "stages": {},
    }
    video_items: list[dict[str, Any]] = []
    state_path = Path(config.four_wall_state)
    if config.from_scratch:
        four_out = out_dir / "four_walls"
        four_run = _run_command(four_wall_command(config, four_out), four_out, timeout_sec=None)
        four_report_path = four_out / "standard_build_report.json"
        four_report = _read_json(four_report_path)
        four_success = bool(four_report and (four_report.get("final") or {}).get("success") and four_run["returncode"] == 0)
        if four_success:
            latest = str((four_report.get("final") or {}).get("latest_stable_state", "") or "")
            if latest:
                state_path = Path(latest)
        report["stages"]["four_walls"] = {
            "success": four_success,
            "run": four_run,
            "report": str(four_report_path),
            "state": str(state_path) if four_success else "",
        }
        video_items.extend(_video_items_from_four_wall(four_report))
        if not four_success:
            report["success"] = False
            report["failed_stage"] = "four_walls"
            report["finished_at"] = datetime.now().isoformat(timespec="seconds")
            write_full_video_manifest(out_dir, video_items)
            (out_dir / "full_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
    else:
        report["stages"]["four_walls"] = {
            "success": state_path.exists(),
            "used_existing_state": str(state_path),
        }
        if not state_path.exists():
            report["success"] = False
            report["failed_stage"] = "four_wall_state_missing"
            report["finished_at"] = datetime.now().isoformat(timespec="seconds")
            (out_dir / "full_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report

    triangle_attempts: list[dict[str, Any]] = []
    triangle_out = out_dir / "second_layer_triangles"
    triangle_run: dict[str, Any] = {}
    triangle_summary_path = triangle_out / "triangle_wall_summary.json"
    triangle_summary: dict[str, Any] | None = None
    triangle_success = False
    max_triangle_attempts = max(int(config.triangle_case_retries), 0) + 1
    for attempt_index in range(max_triangle_attempts):
        attempt_out = triangle_out if attempt_index == 0 else out_dir / f"second_layer_triangles_retry{attempt_index + 1:02d}"
        attempt_roles = _triangle_role_order(attempt_index)
        triangle_run = _run_command(
            triangle_command(config, attempt_out, state_path, roles=attempt_roles),
            attempt_out,
            timeout_sec=float(config.triangle_timeout_sec),
        )
        attempt_summary_path = attempt_out / "triangle_wall_summary.json"
        attempt_summary = _read_json(attempt_summary_path)
        attempt_success = bool(
            attempt_summary and (attempt_summary.get("final") or {}).get("success") and triangle_run["returncode"] == 0
        )
        attempt_video = attempt_out / "triangle_wall_live.mp4"
        triangle_attempts.append(
            {
                "attempt": attempt_index + 1,
                "success": attempt_success,
                "run": triangle_run,
                "summary": str(attempt_summary_path),
                "video": str(attempt_video) if attempt_video.exists() else "",
                "roles": attempt_roles,
                "final": (attempt_summary or {}).get("final", {}),
            }
        )
        if attempt_video.exists():
            video_items.append(
                {
                    "stage": "second_layer_triangles",
                    "role": f"attempt_{attempt_index + 1}",
                    "success": attempt_success,
                    "video": str(attempt_video),
                }
            )
        if attempt_success:
            triangle_success = True
            triangle_out = attempt_out
            triangle_summary_path = attempt_summary_path
            triangle_summary = attempt_summary
            break
        attempt_final = (attempt_summary or {}).get("final", {})
        attempt_state = attempt_out / "triangle_assembly_state.json"
        remaining_roles = _remaining_triangle_roles(attempt_final if isinstance(attempt_final, dict) else {})
        all_role_stable = bool(((attempt_final or {}).get("all_role_stability") or {}).get("success", False))
        if attempt_state.exists() and remaining_roles and all_role_stable:
            continue_out = out_dir / f"second_layer_triangles_continue{attempt_index + 1:02d}"
            continue_run = _run_command(
                triangle_command(config, continue_out, attempt_state, roles=remaining_roles),
                continue_out,
                timeout_sec=float(config.triangle_timeout_sec),
            )
            continue_summary_path = continue_out / "triangle_wall_summary.json"
            continue_summary = _read_json(continue_summary_path)
            continue_success = bool(
                continue_summary
                and (continue_summary.get("final") or {}).get("success")
                and continue_run["returncode"] == 0
            )
            continue_video = continue_out / "triangle_wall_live.mp4"
            triangle_attempts.append(
                {
                    "attempt": f"{attempt_index + 1}.continue",
                    "success": continue_success,
                    "run": continue_run,
                    "summary": str(continue_summary_path),
                    "video": str(continue_video) if continue_video.exists() else "",
                    "roles": remaining_roles,
                    "continued_from_state": str(attempt_state),
                    "final": (continue_summary or {}).get("final", {}),
                }
            )
            if continue_video.exists():
                video_items.append(
                    {
                        "stage": "second_layer_triangles_continue",
                        "role": remaining_roles,
                        "success": continue_success,
                        "video": str(continue_video),
                    }
                )
            if continue_success:
                triangle_success = True
                triangle_out = continue_out
                triangle_summary_path = continue_summary_path
                triangle_summary = continue_summary
                break
    fallback_report: dict[str, Any] | None = None
    if not triangle_success and config.enable_triangle_fallback and config.use_success_profile_fast_path:
        fallback_config = FullBuildConfig(
            out_dir=config.out_dir,
            from_scratch=config.from_scratch,
            four_wall_state=config.four_wall_state,
            four_wall_max_attempts_per_role=config.four_wall_max_attempts_per_role,
            four_wall_timeout_sec=config.four_wall_timeout_sec,
            triangle_case_retries=config.triangle_case_retries,
            triangle_timeout_sec=config.triangle_timeout_sec,
            enable_triangle_fallback=False,
            record_every=config.record_every,
            fps=config.fps,
            use_success_profile_fast_path=False,
        )
        fallback_out = out_dir / "second_layer_triangles_fallback_full_search"
        fallback_run = _run_command(
            triangle_command(fallback_config, fallback_out, state_path),
            fallback_out,
            timeout_sec=float(config.triangle_timeout_sec),
        )
        fallback_summary_path = fallback_out / "triangle_wall_summary.json"
        fallback_summary = _read_json(fallback_summary_path)
        fallback_success = bool(
            fallback_summary and (fallback_summary.get("final") or {}).get("success") and fallback_run["returncode"] == 0
        )
        fallback_report = {
            "success": fallback_success,
            "run": fallback_run,
            "summary": str(fallback_summary_path),
            "video": str(fallback_out / "triangle_wall_live.mp4") if (fallback_out / "triangle_wall_live.mp4").exists() else "",
            "final": (fallback_summary or {}).get("final", {}),
        }
        fallback_video = fallback_out / "triangle_wall_live.mp4"
        if fallback_video.exists():
            video_items.append(
                {
                    "stage": "second_layer_triangles_fallback_full_search",
                    "role": "all",
                    "success": fallback_success,
                    "video": str(fallback_video),
                }
            )
        if fallback_success:
            triangle_success = True
            triangle_out = fallback_out
            triangle_run = fallback_run
            triangle_summary_path = fallback_summary_path
            triangle_summary = fallback_summary
    triangle_video = triangle_out / "triangle_wall_live.mp4"
    report["stages"]["second_layer_triangles"] = {
        "success": triangle_success,
        "run": triangle_run,
        "summary": str(triangle_summary_path),
        "video": str(triangle_video) if triangle_video.exists() else "",
        "final": (triangle_summary or {}).get("final", {}),
        "attempts": triangle_attempts,
        "fallback": fallback_report,
    }
    report["success"] = bool(triangle_success)
    report["failed_stage"] = "" if triangle_success else "second_layer_triangles"
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_full_video_manifest(out_dir, video_items)
    if triangle_success:
        update_profile_cache(out_dir / "profile_cache.json", [triangle_summary_path])
    (out_dir / "full_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
