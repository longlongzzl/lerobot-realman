from __future__ import annotations

import html
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import load_json, to_jsonable

CALIBRATION_REPORT_SCHEMA_VERSION = 1
REPORT_TRANSFORM_KEYS = ("T_R_Cg", "T_R_P", "T_E_Cw")


def _command_string(command: str | list[str] | tuple[str, ...] | None) -> str:
    if command is None:
        return shlex.join(sys.argv)
    if isinstance(command, str):
        return command
    return shlex.join([str(item) for item in command])


def _first_present(payload: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


def _as_int_or_default(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _infer_accepted(report: dict[str, Any], metrics: dict[str, Any] | None) -> bool:
    if "accepted" in report:
        return bool(report["accepted"])
    status = str(report.get("status", "")).lower()
    if status in {"pass", "passed", "ok", "accepted", "success"}:
        return True
    if status in {"fail", "failed", "rejected", "error"}:
        return False
    if metrics is not None and "accepted" in metrics:
        return bool(metrics["accepted"])
    if report.get("failure_reasons"):
        return False
    return True


def normalize_calibration_report(
    report: dict[str, Any],
    *,
    run_kind: str,
    command: str | list[str] | tuple[str, ...] | None = None,
    config_path: str | Path | None = None,
    frames_total: int | None = None,
    frames_used: int | None = None,
    accepted: bool | None = None,
    rejection_reason: str | None = None,
    transforms: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a report with the shared calibration schema added as aliases.

    Existing route-specific fields are preserved. The shared keys make runtime
    lookup and reports consistent across classical board, anchored board, joint,
    and visual-refine calibration routes.
    """

    normalized = dict(report)
    normalized["schema_version"] = int(normalized.get("schema_version", CALIBRATION_REPORT_SCHEMA_VERSION))
    normalized["run_kind"] = str(normalized.get("run_kind") or run_kind)
    normalized["created_at"] = str(
        normalized.get("created_at") or datetime.now().astimezone().isoformat(timespec="seconds")
    )
    normalized["command"] = _command_string(command if command is not None else normalized.get("command"))
    existing_config_path = normalized.get("config_path", "")
    normalized["config_path"] = (
        "" if existing_config_path is None else str(existing_config_path)
    ) if config_path is None else str(Path(config_path).expanduser())

    if frames_total is None:
        frames_total = _first_present(normalized, ("frames_total", "total_frames", "frames", "samples"), 0)
    if frames_used is None:
        frames_used = _first_present(
            normalized,
            ("frames_used", "accepted_frames", "main_board_valid", "paired_frames"),
            frames_total,
        )
    normalized["frames_total"] = _as_int_or_default(frames_total)
    normalized["frames_used"] = _as_int_or_default(frames_used)

    metrics_payload = metrics if metrics is not None else normalized.get("metrics", {})
    if metrics_payload is None:
        metrics_payload = {}
    if not isinstance(metrics_payload, dict):
        metrics_payload = {"value": metrics_payload}
    normalized["metrics"] = metrics_payload

    if accepted is None:
        accepted = _infer_accepted(normalized, metrics_payload)
    normalized["accepted"] = bool(accepted)
    if rejection_reason is None:
        reasons = normalized.get("failure_reasons")
        if isinstance(reasons, list) and reasons:
            rejection_reason = "; ".join(str(item) for item in reasons)
        else:
            rejection_reason = str(normalized.get("rejection_reason", ""))
    normalized["rejection_reason"] = str(rejection_reason or "")

    transform_payload = {key: None for key in REPORT_TRANSFORM_KEYS}
    existing_transforms = normalized.get("transforms")
    if isinstance(existing_transforms, dict):
        for key in REPORT_TRANSFORM_KEYS:
            if key in existing_transforms:
                transform_payload[key] = existing_transforms[key]
    if transforms:
        for key in REPORT_TRANSFORM_KEYS:
            if key in transforms:
                transform_payload[key] = transforms[key]
    normalized["transforms"] = transform_payload
    return normalized


def _format_value(value: Any) -> str:
    value = to_jsonable(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (int, bool)) or value is None:
        return str(value)
    if isinstance(value, str):
        return html.escape(value)
    return html.escape(str(value))


def _summary_table(summary: dict[str, Any]) -> str:
    rows = []
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            continue
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{_format_value(value)}</td>"
            "</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _records_table(records: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not records:
        return "<p class=\"empty\">No records.</p>"
    if columns is None:
        columns = []
        for record in records:
            for key in record.keys():
                key = str(key)
                if key not in columns:
                    columns.append(key)
    header = "".join(f"<th>{html.escape(str(key))}</th>" for key in columns)
    rows = []
    for record in records:
        cells = []
        for key in columns:
            cells.append(f"<td>{_format_value(record.get(key))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + f"<tr>{header}</tr>" + "".join(rows) + "</table>"


def _preformatted(value: Any) -> str:
    value = to_jsonable(value)
    try:
        text = json.dumps(value, indent=2, ensure_ascii=False)
    except TypeError:
        text = str(value)
    return html.escape(text)


def _section_block(heading: str, content: Any) -> str:
    return (
        f"<h2>{html.escape(str(heading))}</h2>"
        "<pre>"
        f"{_preformatted(content)}"
        "</pre>"
    )


def _auto_sections(summary: dict[str, Any], explicit_headings: set[str]) -> list[tuple[str, Any]]:
    sections: list[tuple[str, Any]] = []
    candidates = [
        ("transforms", "Transforms"),
        ("metrics", "Metrics"),
        ("outliers", "Outliers"),
        ("outlier_frames", "Outlier Frames"),
        ("reprojection_metrics", "Reprojection Metrics"),
        ("transform_deltas", "Transform Deltas"),
    ]
    for key, heading in candidates:
        if heading in explicit_headings:
            continue
        value = summary.get(key)
        if value in (None, {}, []):
            continue
        sections.append((heading, value))
    return sections


def _image_cards(run_dir: Path, image_dirs: list[str], max_images_per_dir: int) -> str:
    cards = []
    for image_dir in image_dirs:
        root = run_dir / image_dir
        if not root.exists():
            continue
        for path in sorted(root.glob("*.png"))[:max_images_per_dir]:
            rel = path.relative_to(run_dir).as_posix()
            cards.append(
                "<figure>"
                f"<img src=\"{html.escape(rel)}\" loading=\"lazy\">"
                f"<figcaption>{html.escape(rel)}</figcaption>"
                "</figure>"
            )
    if not cards:
        return ""
    return "<div class=\"grid\">" + "".join(cards) + "</div>"


def write_html_report(
    output_path: str | Path,
    *,
    title: str,
    summary: dict[str, Any],
    tables: list[tuple[str, list[dict[str, Any]], list[str] | None]] | None = None,
    sections: list[tuple[str, Any]] | None = None,
    image_dirs: list[str] | None = None,
    max_images_per_dir: int = 24,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = output_path.parent
    body = [
        f"<h1>{html.escape(title)}</h1>",
        _summary_table(summary),
    ]
    for heading, records, columns in tables or []:
        body.append(f"<h2>{html.escape(str(heading))}</h2>")
        body.append(_records_table(records, columns))
    explicit_sections = sections or []
    explicit_headings = {str(heading) for heading, _ in explicit_sections}
    for heading, content in [*_auto_sections(summary, explicit_headings), *explicit_sections]:
        body.append(_section_block(str(heading), content))
    if image_dirs:
        for image_dir in image_dirs:
            cards = _image_cards(run_dir, [image_dir], int(max_images_per_dir))
            if cards:
                body.append(f"<h2>{html.escape(str(image_dir))}</h2>")
                body.append(cards)
    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172026; background: #f7f8fa; }}
    h1 {{ font-size: 28px; margin: 0 0 16px; }}
    h2 {{ font-size: 18px; margin: 28px 0 12px; }}
    table {{ border-collapse: collapse; min-width: 420px; background: #fff; border: 1px solid #d8dee4; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eaeef2; vertical-align: top; }}
    th {{ width: 230px; color: #4d5b66; background: #f0f3f5; }}
    pre {{ overflow: auto; white-space: pre-wrap; background: #fff; border: 1px solid #d8dee4; padding: 12px; border-radius: 6px; }}
    .empty {{ color: #687782; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #d8dee4; border-radius: 6px; overflow: hidden; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ font-size: 12px; color: #4d5b66; padding: 6px 8px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
""".format(title=html.escape(title), body="\n".join(body))
    output_path.write_text(page, encoding="utf-8")
    return output_path


def write_report_from_json(
    run_dir: str | Path,
    json_name: str,
    *,
    title: str,
    image_dirs: list[str] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    summary = load_json(run_dir / json_name)
    return write_html_report(
        run_dir / "report.html",
        title=title,
        summary=summary,
        sections=[("Full JSON", summary)],
        image_dirs=image_dirs,
    )
