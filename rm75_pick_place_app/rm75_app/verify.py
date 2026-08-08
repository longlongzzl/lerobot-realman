from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .commands import direct_pick_command, roof_assembly_pick_command, sam6d_pick_command, web_command, wrist_refined_pick_command
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
        ("direct module", APP_ROOT / "rm75_app" / "runtime" / "direct_pre_place.py"),
        ("wrist module", APP_ROOT / "rm75_app" / "runtime" / "wrist_refined_pick_place.py"),
        ("roof module", APP_ROOT / "rm75_app" / "runtime" / "roof_assembly_pick_place.py"),
        ("wrist relation adapter", APP_ROOT / "rm75_app" / "perception" / "wrist_relation.py"),
        ("wrist relation cli", APP_ROOT / "rm75_app" / "perception" / "wrist_gripper_object_relation.py"),
        ("sam6d module", APP_ROOT / "rm75_app" / "runtime" / "sam6d_pick_place.py"),
        ("tabletop refine module", APP_ROOT / "rm75_app" / "runtime" / "tabletop_pose_refine.py"),
        ("tabletop refine entrypoint", APP_ROOT / "rm75_app" / "entrypoints" / "tabletop_pose_refine.py"),
        ("web module", APP_ROOT / "rm75_app" / "web" / "control_panel.py"),
        ("llm module", APP_ROOT / "rm75_app" / "llm" / "orchestrator.py"),
        ("core contracts", APP_ROOT / "rm75_app" / "core" / "contracts.py"),
        ("task adapters", APP_ROOT / "rm75_app" / "tasks"),
        ("task pipeline", APP_ROOT / "rm75_app" / "pipeline" / "compiler.py"),
        ("pick-place runner", APP_ROOT / "rm75_app" / "pickplace" / "runner.py"),
        ("pick-place defaults", APP_ROOT / "rm75_app" / "pickplace" / "config.py"),
    ]
    for name, path in required_paths:
        checks.append(Check(name, Path(path).exists(), str(path)))

    for name, cmd in (
        ("direct command local", direct_pick_command()),
        ("wrist command local", wrist_refined_pick_command()),
        ("roof command local", roof_assembly_pick_command()),
        ("sam6d command local", sam6d_pick_command()),
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
        APP_ROOT / "rm75_app" / "runtime" / "direct_pre_place.py",
        APP_ROOT / "rm75_app" / "runtime" / "sam6d_pick_place.py",
        APP_ROOT / "rm75_app" / "runtime" / "targeted_place.py",
        APP_ROOT / "rm75_app" / "runtime" / "targeted_curobo.py",
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
