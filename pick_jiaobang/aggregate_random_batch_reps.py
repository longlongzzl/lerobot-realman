#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: aggregate_random_batch_reps.py <run_root>")
    root = Path(sys.argv[1]).expanduser().resolve()
    records = []
    for path in sorted(root.glob("rep_*/scene_results.jsonl")):
        rep = path.parent.name
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item["rep"] = rep
            records.append(item)

    flat_results = []
    for scene in records:
        for result in scene.get("results", []) or []:
            row = dict(result)
            row["rep"] = scene.get("rep")
            row["scene_file"] = scene.get("scene_file")
            row["scene_mode"] = scene.get("mode")
            row["scene_elapsed_s"] = scene.get("elapsed_s")
            flat_results.append(row)

    object_success = sum(1 for row in flat_results if bool(row.get("success")))
    scene_success = sum(1 for scene in records if bool((scene.get("test_summary") or {}).get("success")))

    object_counts: dict[str, Counter] = defaultdict(Counter)
    failure_codes: Counter = Counter()
    for row in flat_results:
        object_name = str(row.get("object_name", ""))
        key = "success" if bool(row.get("success")) else str(row.get("failure_reason_code") or "failure")
        object_counts[object_name][key] += 1
        if key != "success":
            failure_codes[key] += 1

    scene_elapsed = [float(scene.get("elapsed_s")) for scene in records if scene.get("elapsed_s") is not None]
    rep_elapsed: dict[str, float] = defaultdict(float)
    rep_scene_count: Counter = Counter()
    for scene in records:
        rep = str(scene.get("rep"))
        rep_elapsed[rep] += float(scene.get("elapsed_s") or 0.0)
        rep_scene_count[rep] += 1

    summary = {
        "root": str(root),
        "generated_at": time.strftime("%F %T"),
        "scene_runs": len(records),
        "expected_scene_runs": 64 * 3,
        "scene_success": scene_success,
        "scene_success_rate": scene_success / len(records) if records else 0.0,
        "object_results": len(flat_results),
        "object_success": object_success,
        "object_success_rate": object_success / len(flat_results) if flat_results else 0.0,
        "elapsed_s_total": sum(scene_elapsed),
        "elapsed_s_avg_scene": statistics.mean(scene_elapsed) if scene_elapsed else 0.0,
        "elapsed_s_median_scene": statistics.median(scene_elapsed) if scene_elapsed else 0.0,
        "elapsed_s_max_scene": max(scene_elapsed) if scene_elapsed else 0.0,
        "rep_elapsed_s": dict(sorted(rep_elapsed.items())),
        "rep_scene_count": dict(sorted(rep_scene_count.items())),
        "failure_codes": dict(failure_codes),
        "object_counts": {name: dict(counter) for name, counter in sorted(object_counts.items())},
    }
    (root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Aggregate Random Batch Summary",
        "",
        f"- Root: `{root}`",
        f"- Scene runs: {len(records)}/192",
        (
            f"- Scene success: {scene_success}/{len(records)} "
            f"({summary['scene_success_rate'] * 100.0:.1f}%)"
            if records
            else "- Scene success: n/a"
        ),
        (
            f"- Object success: {object_success}/{len(flat_results)} "
            f"({summary['object_success_rate'] * 100.0:.1f}%)"
            if flat_results
            else "- Object success: n/a"
        ),
        f"- Avg scene elapsed: {summary['elapsed_s_avg_scene']:.1f}s",
        f"- Median scene elapsed: {summary['elapsed_s_median_scene']:.1f}s",
        f"- Max scene elapsed: {summary['elapsed_s_max_scene']:.1f}s",
        "",
        "## Per Rep",
        "",
        "| rep | scenes | elapsed_s |",
        "|---|---:|---:|",
    ]
    for rep in sorted(rep_scene_count):
        lines.append(f"| {rep} | {rep_scene_count[rep]} | {rep_elapsed[rep]:.1f} |")

    lines.extend(["", "## Per Object", "", "| object | success | failures |", "|---|---:|---:|"])
    for object_name, counter in sorted(object_counts.items()):
        ok_count = counter.get("success", 0)
        fail_count = sum(count for key, count in counter.items() if key != "success")
        lines.append(f"| {object_name} | {ok_count} | {fail_count} |")

    lines.extend(["", "## Failure Codes", ""])
    if failure_codes:
        for code, count in failure_codes.most_common():
            lines.append(f"- `{code}`: {count}")
    else:
        lines.append("None observed.")
    (root / "aggregate_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
