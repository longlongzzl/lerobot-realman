from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


DEFAULT_QWEN_VL_BASE_URL = "https://llm-q7nh1xonye3vc6id.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_VL_MODEL = "qwen3.8-max"
DEFAULT_QWEN_VL_KEY_ENV = "RM75_VLM_API_KEY"
DEFAULT_QWEN_VL_PROXY = os.environ.get("RM75_VLM_PROXY", "http://127.0.0.1:7897").strip()


def load_env_file(path: str | Path) -> None:
    """Load a small dotenv file without adding a runtime dependency."""
    env_path = Path(path).expanduser()
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def _extract_json_object(text: str) -> dict[str, Any]:
    content = str(text or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Qwen-VL response did not contain a JSON object")


def _normalized_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        box = [float(number) for number in value[:4]]
    except (TypeError, ValueError):
        return None
    scale = 1000.0 if max(abs(number) for number in box) > 1.5 else 1.0
    box = [min(1.0, max(0.0, number / scale)) for number in box]
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(number, 6) for number in box]


def normalize_inventory(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.get("objects") or []):
        if not isinstance(raw, dict):
            continue
        phrase = str(raw.get("noun_phrase") or raw.get("label") or "").strip().strip(".")
        bbox = _normalized_bbox(raw.get("bbox_normalized") or raw.get("bbox_2d") or raw.get("bbox"))
        if not phrase or bbox is None:
            continue
        aliases = []
        for alias in raw.get("aliases") or []:
            clean = str(alias).strip().strip(".")
            if clean and clean.lower() != phrase.lower() and clean not in aliases:
                aliases.append(clean)
        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        normalized.append(
            {
                "temporary_id": str(raw.get("temporary_id") or f"vlm_{index + 1:02d}"),
                "noun_phrase": phrase,
                "aliases": aliases[:3],
                "bbox_normalized": bbox,
                "attributes": [str(value) for value in (raw.get("attributes") or []) if str(value).strip()][:8],
                "possible_known_asset": raw.get("possible_known_asset"),
                "confidence": confidence,
            }
        )
    return {"objects": normalized, "scene_summary": str(payload.get("scene_summary") or "").strip()}


def inventory_to_sam3_items(
    inventory: dict[str, Any],
    width: int,
    height: int,
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, obj in enumerate(inventory.get("objects") or []):
        bbox = obj.get("bbox_normalized")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        pixel_box = [
            offset_x + bbox[0] * width,
            offset_y + bbox[1] * height,
            offset_x + bbox[2] * width,
            offset_y + bbox[3] * height,
        ]
        items.append(
            {
                "id": str(obj.get("temporary_id") or f"vlm_{index + 1:02d}"),
                "object_name": None,
                "mode": "text_box",
                "prompt": str(obj["noun_phrase"]),
                "box": pixel_box,
                "select_box": pixel_box,
            }
        )
    return items


def merge_inventory_metadata(results: Iterable[dict[str, Any]], inventory: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = {str(item.get("temporary_id")): item for item in inventory.get("objects") or []}
    output = []
    for raw in results:
        item = dict(raw)
        vlm = metadata.get(str(item.get("id")))
        if vlm:
            item["vlm"] = dict(vlm)
        output.append(item)
    return output


@dataclass(frozen=True)
class RemoteVLMConfig:
    base_url: str = DEFAULT_QWEN_VL_BASE_URL
    model: str = DEFAULT_QWEN_VL_MODEL
    api_key_env: str = DEFAULT_QWEN_VL_KEY_ENV
    timeout_s: float = 90.0
    max_image_size: int = 1280
    proxy_url: str = DEFAULT_QWEN_VL_PROXY


class RemoteQwenVLProvider:
    def __init__(self, config: RemoteVLMConfig, *, env_file: str | Path | None = None):
        self.config = config
        if env_file is not None:
            load_env_file(env_file)

    def _api_key(self) -> str:
        value = str(os.environ.get(self.config.api_key_env) or os.environ.get("DASHSCOPE_API_KEY") or "").strip()
        if not value:
            raise RuntimeError(f"VLM API key is missing; set {self.config.api_key_env}")
        return value

    def _image_data_url(self, image_path: str | Path) -> tuple[str, int, int]:
        image = Image.open(Path(image_path).expanduser()).convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > int(self.config.max_image_size):
            scale = float(self.config.max_image_size) / float(longest)
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}", width, height

    @staticmethod
    def _prompt(known_assets: dict[str, str]) -> str:
        known_text = "\n".join(f"- {name}: {prompt}" for name, prompt in sorted(known_assets.items()))
        return f"""Analyze this fixed-camera RGB image of a robot tabletop.
List every visible, independently movable physical object on the working table, including unfamiliar objects.
Do not list the table, robot arm, gripper, walls, shadows, printed pictures, or parts of another object.
Give one entry per physical instance, even when multiple objects have the same category.
Use a short concrete English noun phrase suitable as a SAM3 segmentation prompt.
Return bbox_normalized as [x1,y1,x2,y2] in the 0..1000 coordinate system.
possible_known_asset must be one of the provided internal asset IDs or null; do not force a match.

Known asset catalog:
{known_text}

Return JSON only, with exactly this shape:
{{"scene_summary":"...","objects":[{{"temporary_id":"obj_01","noun_phrase":"orange carrot","aliases":["carrot"],"bbox_normalized":[0,0,1000,1000],"attributes":["orange"],"possible_known_asset":null,"confidence":0.0}}]}}"""

    def inventory(self, image_path: str | Path, known_assets: dict[str, str]) -> dict[str, Any]:
        image_url, width, height = self._image_data_url(image_path)
        body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": self._prompt(known_assets)},
                    ],
                }
            ],
            "stream": False,
            "max_tokens": 4096,
            "enable_thinking": False,
        }
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            if self.config.proxy_url:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler(
                        {"http": self.config.proxy_url, "https": self.config.proxy_url}
                    )
                )
            else:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=float(self.config.timeout_s)) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Qwen-VL HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qwen-VL request failed: {exc.reason}") from exc
        choices = response_payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"Qwen-VL returned no choices: {response_payload}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
        inventory = normalize_inventory(_extract_json_object(str(content or "")))
        inventory.update(
            {
                "provider": "qwen_vl",
                "model": self.config.model,
                "image_path": str(Path(image_path).expanduser()),
                "image_width": width,
                "image_height": height,
                "usage": response_payload.get("usage"),
            }
        )
        return inventory
