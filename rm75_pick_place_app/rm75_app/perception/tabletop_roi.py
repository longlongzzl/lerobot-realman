from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]
) -> bool:
    eps = 1e-9
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    return ab_c * ab_d < -eps and cd_a * cd_b < -eps


def normalize_five_point_polygon(points: Iterable[Any]) -> list[list[float]]:
    values = list(points)
    if len(values) != 5:
        raise ValueError("桌面 ROI 必须正好包含 5 个点")
    normalized: list[tuple[float, float]] = []
    for index, point in enumerate(values):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError(f"ROI 第 {index + 1} 个点格式错误")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ROI 第 {index + 1} 个点不是数字") from exc
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"ROI 第 {index + 1} 个点超出图像范围")
        normalized.append((x, y))

    area2 = sum(
        normalized[index][0] * normalized[(index + 1) % 5][1]
        - normalized[(index + 1) % 5][0] * normalized[index][1]
        for index in range(5)
    )
    if abs(area2) * 0.5 < 0.01:
        raise ValueError("桌面 ROI 面积太小，请沿桌面边界重新点选")
    edges = [(index, (index + 1) % 5) for index in range(5)]
    for first, (a_idx, b_idx) in enumerate(edges):
        for c_idx, d_idx in edges[first + 1 :]:
            if len({a_idx, b_idx, c_idx, d_idx}) < 4:
                continue
            if _segments_intersect(normalized[a_idx], normalized[b_idx], normalized[c_idx], normalized[d_idx]):
                raise ValueError("桌面 ROI 边界发生交叉，请按顺时针或逆时针顺序点选")
    return [[round(x, 6), round(y, 6)] for x, y in normalized]


def load_tabletop_roi(path: str | Path) -> dict[str, Any] | None:
    roi_path = Path(path).expanduser()
    try:
        payload = json.loads(roi_path.read_text(encoding="utf-8"))
        payload["points_normalized"] = normalize_five_point_polygon(payload.get("points_normalized") or [])
        return payload
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, TypeError):
        return None


def save_tabletop_roi(
    path: str | Path,
    points: Iterable[Any],
    *,
    camera_serial: str | None = None,
    image_size: Iterable[Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_five_point_polygon(points)
    size = list(image_size or [])
    payload = {
        "schema_version": 1,
        "kind": "five_point_polygon",
        "points_normalized": normalized,
        "camera_serial": str(camera_serial or "") or None,
        "image_size": [int(size[0]), int(size[1])] if len(size) >= 2 else None,
        "updated_at": time.time(),
    }
    roi_path = Path(path).expanduser()
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = roi_path.with_suffix(roi_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, roi_path)
    return payload


def crop_polygon_roi(
    image_path: str | Path,
    points: Iterable[Any],
    output_path: str | Path,
) -> dict[str, Any]:
    normalized = normalize_five_point_polygon(points)
    image = Image.open(Path(image_path).expanduser()).convert("RGB")
    width, height = image.size
    pixels = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in normalized]
    x1 = max(0, min(point[0] for point in pixels))
    y1 = max(0, min(point[1] for point in pixels))
    x2 = min(width, max(point[0] for point in pixels) + 1)
    y2 = min(height, max(point[1] for point in pixels) + 1)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(pixels, fill=255)
    canvas = Image.new("RGB", image.size, (96, 96, 96))
    canvas.paste(image, mask=mask)
    crop = canvas.crop((x1, y1, x2, y2))
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    crop.save(destination, format="JPEG", quality=92)
    return {
        "image_path": str(destination),
        "original_width": width,
        "original_height": height,
        "crop_width": crop.width,
        "crop_height": crop.height,
        "offset_x": x1,
        "offset_y": y1,
        "points_normalized": normalized,
        "polygon_pixels": [[int(x), int(y)] for x, y in pixels],
    }


def point_in_polygon(x: float, y: float, points: Iterable[Any]) -> bool:
    polygon = [(float(point[0]), float(point[1])) for point in points]
    inside = False
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        xi, yi = polygon[current]
        xj, yj = polygon[previous]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        if intersects:
            inside = not inside
        previous = current
    return inside
