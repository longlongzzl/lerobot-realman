from __future__ import annotations

from pathlib import Path

from rm75_app.paths import RUNTIME_DIR


def sam6d_provider_args(*, mask_mode: str = "sam3_text", object_names: list[str] | None = None) -> list[str]:
    names = object_names or ["lvmukuai", "carriot", "shuazi", "hongshupian", "gluestick", "bi", "tennis", "desk", "bitong"]
    return [
        "--output-root",
        str(RUNTIME_DIR / "sam6d_grasp_scene_runs"),
        "--object-names",
        *names,
        "--mask-mode",
        str(mask_mode),
        "--sam3-provider-script",
        str(Path(__file__).resolve().with_name("sam3_mask_provider.py")),
        "--pem-feature-cache-root",
        str(RUNTIME_DIR / "sam6d_pem_feature_cache"),
        "--template-cache-root",
        str(RUNTIME_DIR / "sam6d_template_cache"),
    ]
