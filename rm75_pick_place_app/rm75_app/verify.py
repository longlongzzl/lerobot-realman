from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .commands import direct_pick_command, web_command
from .core.contracts import TaskRequest
from .launch import local_runtime
from .paths import APP_ROOT, DEFAULT_CUROBO_CFG, DEFAULT_RM75_URDF, MESH_DIR, RUNTIME_DIR, TEST_SCENE_DIR
from .pickplace import PICKPLACE_BACKENDS, PICKPLACE_LAYERS
from .pipeline import TaskPipeline
from .tasks import list_task_adapters


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _inside(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def run_checks() -> list[Check]:
    checks: list[Check] = []
    required_paths = [
        ("app root", APP_ROOT),
        ("curobo cfg", DEFAULT_CUROBO_CFG),
        ("robot urdf", DEFAULT_RM75_URDF),
        ("meshs", MESH_DIR),
        ("test scenes", TEST_SCENE_DIR),
        ("runtime data", RUNTIME_DIR),
        ("wrist relation adapter", APP_ROOT / "rm75_app" / "perception" / "wrist_relation.py"),
        ("wrist relation cli", APP_ROOT / "rm75_app" / "perception" / "wrist_gripper_object_relation.py"),
        ("sam6d pose provider", APP_ROOT / "rm75_app" / "perception" / "sam6d_pose_provider.py"),
        ("rrtrack module", APP_ROOT / "rm75_app" / "runtime" / "rrtrack_pose_tracking.py"),
        ("rrtrack bank builder", APP_ROOT / "rm75_app" / "runtime" / "rrtrack_build_bank.py"),
        ("rrtrack all-object bank builder", APP_ROOT / "rm75_app" / "runtime" / "rrtrack_build_all_banks.py"),
        ("rrtrack core", APP_ROOT / "rm75_app" / "perception" / "rrtrack" / "tracker.py"),
        ("rrtrack CUTIE adapter", APP_ROOT / "rm75_app" / "perception" / "rrtrack" / "cutie_adapter.py"),
        ("rrtrack FoundationPose adapter", APP_ROOT / "rm75_app" / "perception" / "rrtrack" / "foundationpose_adapter.py"),
        ("rrtrack scene registry", APP_ROOT / "rm75_app" / "perception" / "rrtrack" / "scene_registry.py"),
        ("rrtrack SAM3 fallback", APP_ROOT / "rm75_app" / "perception" / "rrtrack" / "sam3_relocalizer.py"),
        ("openworld geometry runtime", APP_ROOT / "rm75_app" / "runtime" / "openworld_geometry.py"),
        ("openworld geometry session", APP_ROOT / "rm75_app" / "perception" / "openworld_geometry" / "session.py"),
        ("dynamic geometry cuRobo world", APP_ROOT / "rm75_app" / "planning" / "dynamic_geometry_world.py"),
        ("planning contracts", APP_ROOT / "rm75_app" / "planning" / "contracts.py"),
        ("Curobo2 backend", APP_ROOT / "rm75_app" / "planning" / "backends" / "curobo2.py"),
        ("pick-place coordinator", APP_ROOT / "rm75_app" / "pickplace" / "coordinator.py"),
        ("manipulation task protocol", APP_ROOT / "rm75_app" / "tasks" / "manipulation_plan.py"),
        ("multi-object task executor", APP_ROOT / "rm75_app" / "orchestration" / "multi_object_executor.py"),
        ("pick-place atom task builder", APP_ROOT / "rm75_app" / "pickplace" / "atom_task_builder.py"),
        ("pick-place multi-object adapter", APP_ROOT / "rm75_app" / "pickplace" / "multi_object_adapter.py"),
        ("ManiSkill task bridge", APP_ROOT / "rm75_app" / "execution" / "maniskill_task_bridge.py"),
        ("three-gate task validation", APP_ROOT / "rm75_app" / "validation" / "three_gate.py"),
        ("ManiSkill scene preview", APP_ROOT / "rm75_app" / "runtime" / "maniskill_scene_preview.py"),
        ("multi-object ManiSkill environment", APP_ROOT / "rm75_app" / "simulation" / "maniskill_multi_object_env.py"),
        ("Curobo2 runtime", APP_ROOT / "rm75_app" / "runtime" / "curobo2_pick_place.py"),
        ("Curobo2 sim replay", APP_ROOT / "rm75_app" / "runtime" / "curobo2_sim_replay.py"),
        ("portable trajectory executor", APP_ROOT / "rm75_app" / "execution" / "trajectory_executor.py"),
        ("tabletop refine module", APP_ROOT / "rm75_app" / "runtime" / "tabletop_pose_refine.py"),
        ("tabletop refine entrypoint", APP_ROOT / "rm75_app" / "entrypoints" / "tabletop_pose_refine.py"),
        ("web module", APP_ROOT / "rm75_app" / "web" / "control_panel.py"),
        ("scene workbench", APP_ROOT / "rm75_app" / "web" / "scene_workbench.py"),
        ("llm module", APP_ROOT / "rm75_app" / "llm" / "orchestrator.py"),
        ("llm complete-manifest runner", APP_ROOT / "rm75_app" / "runtime" / "llm_manifest_execution.py"),
        ("core contracts", APP_ROOT / "rm75_app" / "core" / "contracts.py"),
        ("task adapters", APP_ROOT / "rm75_app" / "tasks"),
        ("task pipeline", APP_ROOT / "rm75_app" / "pipeline" / "compiler.py"),
        ("pick-place runner", APP_ROOT / "rm75_app" / "pickplace" / "runner.py"),
        ("pick-place defaults", APP_ROOT / "rm75_app" / "pickplace" / "config.py"),
    ]
    for name, path in required_paths:
        checks.append(Check(name, Path(path).exists(), str(path)))

    for name, cmd in (
        ("Curobo2 direct alias local", direct_pick_command()),
        ("web command local", web_command()),
    ):
        old_dir_markers = ("pick_" + "jiaobang/", "pick_" + "jiaobang\\")
        bad_parts = [part for part in cmd if any(marker in str(part) for marker in old_dir_markers)]
        checks.append(Check(name, not bad_parts, "bad_parts=" + repr(bad_parts)))

    for mode, backend in PICKPLACE_BACKENDS.items():
        module_path = APP_ROOT / "rm75_app" / (backend.module.replace("rm75_app.", "").replace(".", "/") + ".py")
        checks.append(Check(f"pick-place backend {mode}", module_path.exists(), str(module_path)))

    for layer in PICKPLACE_LAYERS:
        missing_modules = []
        for module in layer.modules:
            module_path = APP_ROOT / "rm75_app" / (module.replace("rm75_app.", "").replace(".", "/") + ".py")
            if not module_path.exists():
                missing_modules.append(str(module_path))
        checks.append(Check(f"pick-place layer {layer.key}", not missing_modules, "missing=" + repr(missing_modules)))

    source_boundary_files = (
        APP_ROOT / "rm75_app" / "runtime" / "curobo2_pick_place.py",
        APP_ROOT / "rm75_app" / "runtime" / "rrtrack_pose_tracking.py",
        APP_ROOT / "rm75_app" / "assets" / "object_specs.py",
    )
    forbidden_source_markers = ("pick_jiaobang", "rm75_lego_snap_place_app", "Beta_demo-codex")
    boundary_hits = []
    for source_file in source_boundary_files:
        text = source_file.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden_source_markers:
            if marker in text:
                boundary_hits.append(f"{source_file.name}:{marker}")
    checks.append(Check("pick-place source boundary", not boundary_hits, "hits=" + repr(boundary_hits)))

    mainline_files = (
        APP_ROOT / "rm75_app" / "planning" / "backends" / "curobo2.py",
        APP_ROOT / "rm75_app" / "pickplace" / "coordinator.py",
        APP_ROOT / "rm75_app" / "runtime" / "curobo2_pick_place.py",
        APP_ROOT / "rm75_app" / "runtime" / "curobo2_sim_replay.py",
        APP_ROOT / "rm75_app" / "execution" / "trajectory_executor.py",
    )
    legacy_import_hits = []
    for source_file in mainline_files:
        text = source_file.read_text(encoding="utf-8", errors="ignore")
        if "runtime.direct_pre_place" in text or "runtime import direct_pre_place" in text:
            legacy_import_hits.append(source_file.name)
    checks.append(Check("Curobo2 mainline excludes 20k legacy executor", not legacy_import_hits, "hits=" + repr(legacy_import_hits)))

    pipeline = TaskPipeline()
    for adapter in list_task_adapters():
        definition = adapter.definition
        try:
            compiled = pipeline.compile(TaskRequest(task=definition.key))
            plan_ok = bool(compiled.plan.steps) and compiled.plan.definition.key == definition.key
            detail = f"mode={compiled.plan.request.mode}, stages={len(compiled.plan.steps)}"
        except Exception as exc:
            plan_ok = False
            detail = repr(exc)
        checks.append(Check(f"task adapter {definition.key}", plan_ok, detail))

    with local_runtime(None):
        from rm75_app.assets import object_specs

        missing_meshes: list[str] = []
        external_meshes: list[str] = []
        for spec_name, spec in object_specs.OBJECT_SPECS.items():
            for field_name in ("mesh_file", "sim_asset_file"):
                value = getattr(spec, field_name, None)
                if not value:
                    continue
                path = Path(value).expanduser()
                if not _inside(path, APP_ROOT):
                    external_meshes.append(f"{spec_name}.{field_name}={path}")
                elif not path.exists():
                    missing_meshes.append(f"{spec_name}.{field_name}={path}")
        checks.append(Check("object_specs meshes local", not external_meshes, "external=" + repr(external_meshes[:8])))
        checks.append(Check("object_specs meshes exist", not missing_meshes, "missing=" + repr(missing_meshes[:8])))

    return checks


def main() -> int:
    checks = run_checks()
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
