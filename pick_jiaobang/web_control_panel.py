#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import psutil
from flask import Flask, Response, jsonify, request, send_file


ROOT = Path(__file__).resolve().parents[1]
PICK_DIR = Path(__file__).resolve().parent
DEFAULT_SAM3_PYTHON = "/home/zhangzhao/anaconda3/envs/sam3/bin/python"
DEFAULT_FOUNDATIONPOSE_PYTHON = "/home/zhangzhao/anaconda3/envs/foundationpose310/bin/python"
DEFAULT_SAM3_CHECKPOINT = "/home/zhangzhao/Downloads/sam3.pt"
DEFAULT_CAMERA_EXTRINSIC_OPENCV = "/home/zhangzhao/Desktop/lerobot-sim2real/results/realman/realman_home/base_camera/camera_extrinsic_opencv.npy"
DEFAULT_GRASP_OBJECTS = ["lvmukuai", "carriot", "shuazi", "hongshupian", "gluestick", "bi", "tennis"]
DEFAULT_TRACKED_OBJECTS = ["desk", "bitong"]
DEFAULT_OBJECTS = DEFAULT_GRASP_OBJECTS + DEFAULT_TRACKED_OBJECTS
SERVER_STARTED_AT = time.time()
latest_perception_result: dict[str, Any] = {}


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in cmd)


def _as_names(value, allowed: list[str], fallback: list[str]) -> list[str]:
    if value is None:
        return list(fallback)
    if isinstance(value, str):
        raw_items = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = []
    allowed_set = set(allowed)
    out = []
    for raw in raw_items:
        name = str(raw).strip()
        if name in allowed_set and name not in out:
            out.append(name)
    return out


def _slot_order_tokens(value) -> list[str]:
    if isinstance(value, str):
        raw_items = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = []
    out = []
    for raw in raw_items:
        token = str(raw).strip()
        if not token:
            continue
        if token.lower().startswith("slot_"):
            token = token.split("_")[-1]
        try:
            slot_idx = int(token)
        except Exception:
            continue
        if 1 <= slot_idx <= 6 and str(slot_idx) not in out:
            out.append(str(slot_idx))
    return out


def _source_slot_map_tokens(value) -> list[str]:
    if isinstance(value, str):
        raw_items = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = []
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        token = str(raw).strip()
        if not token:
            continue
        if ":" in token:
            source, slot = token.split(":", 1)
        elif "=" in token:
            source, slot = token.split("=", 1)
        else:
            continue
        source = source.strip()
        if source == "bi":
            raise ValueError("bi 固定放入 bitong，不支持绑定桌面 slot")
        if source not in DEFAULT_GRASP_OBJECTS:
            continue
        slot_tokens = _slot_order_tokens([slot])
        if not slot_tokens:
            continue
        if source in seen:
            continue
        seen.add(source)
        out.append(f"{source}:{slot_tokens[0]}")
    return out


def build_grasp_command(config: dict[str, Any] | None = None) -> list[str]:
    config = dict(config or {})
    config["reuse_latest_perception"] = True
    objects = _as_names(config.get("objects"), DEFAULT_GRASP_OBJECTS, DEFAULT_GRASP_OBJECTS)
    tracked = _as_names(config.get("tracked_objects"), DEFAULT_TRACKED_OBJECTS, DEFAULT_TRACKED_OBJECTS)
    slot_order = _slot_order_tokens(config.get("slot_order"))
    source_slot_map = _source_slot_map_tokens(config.get("source_slot_map"))
    if not objects:
        raise ValueError("至少选择一个抓取目标")
    fixed_scene_result = config.get("sam6d_fixed_scene_result_file")
    if fixed_scene_result is None and bool(config.get("reuse_latest_perception", False)):
        fixed_scene_result = latest_perception_result.get("result_path")
    if fixed_scene_result is not None:
        fixed_scene_result = str(Path(str(fixed_scene_result)).expanduser())
        if not Path(fixed_scene_result).exists():
            raise ValueError(f"最新分割定位结果不存在: {fixed_scene_result}")
    if fixed_scene_result and bool(config.get("reuse_latest_perception", False)):
        available_objects = list(latest_perception_result.get("pose_found_objects") or latest_perception_result.get("mask_found_objects") or [])
        if available_objects:
            available_set = set(str(name) for name in available_objects)
            requested_objects = list(objects)
            objects = [name for name in objects if name in available_set]
            if not objects:
                raise ValueError(
                    "选择的抓取目标都没有被最近一次 SAM3/SAM6D 成功定位；"
                    f"requested={requested_objects}, available={sorted(available_set)}"
                )
            source_slot_map = [token for token in source_slot_map if token.split(":", 1)[0] in set(objects)]
    cmd = [
        DEFAULT_FOUNDATIONPOSE_PYTHON,
        str(PICK_DIR / "rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place_sam6d.py"),
        "--curobo-rm75-robot-cfg",
        str(PICK_DIR / "curobo_rm75_config/rm75.yml"),
        "--trajectory-preview-sleep",
        "0.08",
        "--cycle-object-names",
        *objects,
        "--tracked-scene-object-names",
        *tracked,
        "--render-mode",
        str(config.get("render_mode") or "human"),
        "--auto-execute",
        "--dry-run-motion-window-scale",
        "1.0",
        "--real-control-hz",
        str(int(config.get("real_control_hz") or 30)),
        "--real-max-delta-per-step",
        str(float(config.get("real_max_delta_per_step") or 0.1)),
    ]
    if bool(config.get("disable_empty_grasp_relocalize", False)):
        cmd.extend(["--no-empty-grasp-relocalize-target", "--empty-grasp-max-relocalize-retries", "0"])
    if bool(config.get("random_targets", False)):
        cmd.extend(["--random-cycle-targets", "--target-selection-order", "random"])
    else:
        cmd.extend(["--target-selection-order", "cycle"])
    if bool(config.get("execute_real", True)):
        cmd.append("--execute-real")
    if slot_order:
        cmd.extend(["--targeted-place-slot-order", *slot_order])
    if source_slot_map:
        cmd.extend(["--targeted-place-source-slot-map", *source_slot_map])
    if not bool(config.get("confirm_segmentation", True)):
        cmd.append("--no-sam6d-confirm-segmentation")
    if fixed_scene_result:
        cmd.extend(["--sam6d-fixed-scene-result-file", fixed_scene_result])
    return cmd


def placement_mapping_text(config: dict[str, Any] | None = None) -> str:
    preview = build_mapping_preview(dict(config or {}))
    parts = []
    for item in preview:
        obj = str(item.get("object") or "")
        dst = str(item.get("destination") or "")
        if obj and dst:
            parts.append(f"{obj}->{dst}")
    return ", ".join(parts)


def build_perception_command(config: dict[str, Any] | None = None) -> list[str]:
    config = dict(config or {})
    object_names = _as_names(config.get("object_names"), DEFAULT_OBJECTS, DEFAULT_OBJECTS)
    if not object_names:
        raise ValueError("至少选择一个分割定位对象")
    cmd = [
        DEFAULT_FOUNDATIONPOSE_PYTHON,
        str(PICK_DIR / "sam6d_groundingdino_pose_provider.py"),
        "--sam6d-root",
        "/home/zhangzhao/PycharmProjects/SAM-6D/SAM-6D",
        "--output-root",
        str(PICK_DIR / "sam6d_grasp_scene_runs"),
        "--object-names",
        *object_names,
        "--mask-mode",
        "sam3_text",
        "--camera-width",
        "640",
        "--camera-height",
        "480",
        "--camera-fps",
        "30",
        "--warmup-frames",
        "30",
        "--camera-extrinsic-opencv-path",
        DEFAULT_CAMERA_EXTRINSIC_OPENCV,
        "--sam3-python",
        DEFAULT_SAM3_PYTHON,
        "--sam3-provider-script",
        str(PICK_DIR / "sam3_mask_provider.py"),
        "--sam3-checkpoint-path",
        DEFAULT_SAM3_CHECKPOINT,
        "--sam3-resolution",
        "1008",
        "--sam3-confidence-threshold",
        "0.35",
        "--sam3-morph-kernel",
        "3",
        "--sam3-max-masks-per-item",
        "1",
        "--pem-feature-cache-root",
        str(PICK_DIR / "sam6d_pem_feature_cache"),
        "--no-post-pem-mask-refine",
        "--post-pem-mask-refine-objects",
        "lvmukuai,carriot,tennis",
        "--post-pem-mask-refine-trigger-px",
        "6.0",
        "--sam3-device",
        "cuda",
        "--no-pem-warmup-during-sam3",
    ]
    if bool(config.get("confirm_segmentation", True)):
        cmd.extend(["--sam3-full-scene-mask-confirm", "--sam3-require-full-scene-masks"])
    else:
        cmd.extend(["--no-sam3-full-scene-mask-confirm", "--no-sam3-show-full-scene-mask-window"])
    return cmd


DEFAULT_GRASP_COMMAND = shell_join(build_grasp_command({"random_targets": True, "execute_real": True}))


app = Flask(__name__)
log_history: deque[dict[str, Any]] = deque(maxlen=1000)
gpu_history: deque[dict[str, Any]] = deque(maxlen=900)
event_clients: list[queue.Queue] = []
event_lock = threading.Lock()


def now_ms() -> int:
    return int(time.time() * 1000)


def format_ms(value) -> str:
    try:
        ms = float(value)
    except Exception:
        return "?"
    if ms >= 1000.0:
        return f"{ms / 1000.0:.2f}s"
    return f"{ms:.0f}ms"


def emit(kind: str, message: str | None = None, **fields) -> None:
    payload = {"ts": now_ms(), "kind": kind}
    if message is not None:
        payload["message"] = message
    payload.update(fields)
    if kind != "gpu":
        log_history.append(payload)
    with event_lock:
        stale = []
        for q in event_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                stale.append(q)
        for q in stale:
            try:
                event_clients.remove(q)
            except ValueError:
                pass


def chinese_summary(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None
    if text.startswith("[fast_chain_ui]"):
        summary = text.split("]", 1)[-1].strip()
        if "；" in summary:
            summary = summary.replace("；", "；\n  ")
        return summary
    if "Traceback" in text or "RuntimeError" in text or "failed:" in text:
        return "检测到错误：" + text[-260:]
    if "[sam3] full-scene text masks:" in text:
        return "SAM3 全场分割完成：" + text
    if "[sam3] full-scene mask overlay:" in text or "[sam3 resident] full-scene mask overlay:" in text:
        return "SAM3 分割可视化已生成：" + text.split(":", 1)[-1].strip()
    if "[sam6d-gdino] ok_count=" in text:
        return "SAM6D 全场位姿完成：" + text
    if "[sam6d-gdino] full-scene PEM overlay:" in text or "[sam6d-resident] full-scene PEM overlay:" in text:
        return "SAM6D PEM 可视化已生成：" + text
    if "[sam6d-resident] ok_count=" in text:
        return "SAM6D 常驻位姿完成：" + text
    if "[prefetch] worker finished" in text:
        return "后台预计算结束：" + text
    if "[prefetch] using cached plan" in text:
        return "命中后台预计算路径：" + text
    if "final success" in text:
        return "本轮最终结果：" + text
    if "[planning_profile] jsonl:" in text:
        return "本次 profile 文件：" + text.split(":", 1)[-1].strip()
    if "[sam6d prefetch] fail-fast" in text:
        return "SAM6D 后台预计算已启用 fail-fast。"
    if "scene_capture" in text and "elapsed_ms" in text:
        return "场景感知阶段记录：" + text[-240:]
    return None


def _safe_log_component(name: str) -> str:
    out = []
    for ch in str(name or "process"):
        out.append(ch if (ch.isalnum() or ch in "-_") else "_")
    return "".join(out).strip("_") or "process"


def _make_process_log_path(process_name: str) -> Path:
    log_dir = PICK_DIR / "web_process_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = _safe_log_component(process_name)
    return log_dir / f"{stamp}_{safe}_{uuid.uuid4().hex[:8]}.log"


def process_line_for_ui(process_name: str, stream_name: str, line: str) -> str | None:
    text = line.strip()
    if not text:
        return None
    # Raw process logs are kept on disk.  The web console only shows explicit
    # concise UI messages; normal stdout/stderr is summarized by chinese_summary().
    for prefix in ("[web_ui]", "[web-ui]"):
        if text.startswith(prefix):
            return text.split("]", 1)[-1].strip()
    return None


class ManagedProcess:
    def __init__(self, name: str) -> None:
        self.name = name
        self.proc: subprocess.Popen | None = None
        self.cmd: list[str] = []
        self.started_at: float | None = None
        self.phase = "idle"
        self.ready = False
        self.last_result: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.last_json: dict[str, Any] | None = None
        self.last_update_at: float | None = None
        self.pending: dict[str, queue.Queue] = {}
        self.raw_log_path: Path | None = None
        self._raw_log_fp = None
        self.lock = threading.Lock()

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
        with self.lock:
            if self.is_running():
                raise RuntimeError(f"{self.name} is already running")
            full_env = os.environ.copy()
            full_env["PYTHONUNBUFFERED"] = "1"
            if env:
                full_env.update(env)
            self.cmd = list(cmd)
            self.phase = "starting"
            self.ready = False
            self.last_result = None
            self.last_error = None
            self.last_json = None
            self.last_update_at = time.time()
            self.raw_log_path = _make_process_log_path(self.name)
            try:
                self._raw_log_fp = self.raw_log_path.open("a", encoding="utf-8", buffering=1)
                self._raw_log_fp.write(f"# process: {self.name}\n")
                self._raw_log_fp.write(f"# started_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                self._raw_log_fp.write(f"# cwd: {cwd}\n")
                self._raw_log_fp.write(f"# command: {shell_join(cmd)}\n\n")
            except Exception:
                self._raw_log_fp = None
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=full_env,
                start_new_session=True,
            )
            self.started_at = time.time()
            log_hint = f"；完整日志：{self.raw_log_path}" if self.raw_log_path else ""
            emit("status", f"{self.name} 已启动 pid={self.proc.pid}{log_hint}")
            threading.Thread(target=self._read_pipe, args=("stdout", self.proc.stdout), daemon=True).start()
            threading.Thread(target=self._read_pipe, args=("stderr", self.proc.stderr), daemon=True).start()
            threading.Thread(target=self._watch, daemon=True).start()

    def _write_raw_log(self, stream_name: str, line: str) -> None:
        fp = self._raw_log_fp
        if fp is None:
            return
        try:
            fp.write(f"[{time.strftime('%H:%M:%S')}] {stream_name}: {line}\n")
        except Exception:
            pass

    def _close_raw_log(self) -> None:
        fp = self._raw_log_fp
        self._raw_log_fp = None
        if fp is None:
            return
        try:
            fp.flush()
            fp.close()
        except Exception:
            pass

    def _read_pipe(self, stream_name: str, pipe) -> None:
        if pipe is None:
            return
        for line in pipe:
            line = line.rstrip("\n")
            self._write_raw_log(stream_name, line)
            self._handle_structured_output(stream_name, line)
            ui_line = process_line_for_ui(self.name, stream_name, line)
            if ui_line:
                emit("process", ui_line, process=self.name, stream=stream_name)
            summary = chinese_summary(line)
            if summary:
                emit("summary", summary, process=self.name)

    def _warmup_success_message(self, result: dict[str, Any]) -> str:
        parts = []
        if "elapsed_ms" in result:
            parts.append(f"加载 {format_ms(result.get('elapsed_ms'))}")
        if "image_warmup_ms" in result:
            parts.append(f"图像预热 {format_ms(result.get('image_warmup_ms'))}")
        if "model_warmup_ms" in result:
            parts.append(f"模型 {format_ms(result.get('model_warmup_ms'))}")
        templates = result.get("template_features")
        if isinstance(templates, dict):
            parts.append(
                "模板 "
                f"{int(templates.get('ok_count', 0))}/{int(templates.get('object_count', 0))} "
                f"{format_ms(templates.get('elapsed_ms'))}"
            )
        detail = "，".join(parts) if parts else "完成"
        return f"{self.name} 热启动成功：{detail}"

    def _handle_structured_output(self, stream_name: str, line: str) -> None:
        if stream_name != "stdout":
            return
        text = line.strip()
        if not text.startswith("{"):
            return
        try:
            payload = json.loads(text)
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        message = None
        pending_q = None
        with self.lock:
            self.last_json = payload
            self.last_update_at = time.time()
            request_id = payload.get("request_id")
            cmd = str(payload.get("cmd") or "")
            if request_id is not None:
                pending_q = self.pending.pop(str(request_id), None)
            if payload.get("ok") is False:
                err = str(payload.get("error") or "unknown error")
                if cmd == "warmup":
                    self.phase = "error"
                    self.ready = False
                    self.last_error = err
                    self.last_result = None
                    message = f"{self.name} 热启动失败：{self.last_error}"
                else:
                    self.last_error = err
                    message = f"{self.name} 常驻命令失败：{err}"
            elif payload.get("event") == "started":
                self.phase = "started"
                self.ready = False
                self.last_error = None
                message = f"{self.name} 常驻进程已启动，等待热启动完成。"
            elif payload.get("cmd") == "warmup" and payload.get("ok") is True:
                result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
                self.phase = "ready"
                self.ready = True
                self.last_result = dict(result)
                self.last_error = None
                message = self._warmup_success_message(result)
            elif payload.get("cmd") == "shutdown" and payload.get("ok") is True:
                self.phase = "stopping"
                self.ready = False
                message = f"{self.name} 收到停止确认。"
        if message:
            emit("hotstart", message, process=self.name, status=self.status())
            emit("summary", message, process=self.name)
        if pending_q is not None:
            try:
                pending_q.put_nowait(payload)
            except queue.Full:
                pass

    def _watch(self) -> None:
        proc = self.proc
        if proc is None:
            return
        code = proc.wait()
        with self.lock:
            if code == 0:
                self.phase = "stopped"
            elif code != 0:
                self.phase = "error"
                self.ready = False
                self.last_error = f"process exited with code {code}"
            self._close_raw_log()
        emit("status", f"{self.name} 已退出 code={code}", process=self.name, returncode=code)

    def send_stdin(self, text: str) -> None:
        with self.lock:
            if not self.is_running() or self.proc is None or self.proc.stdin is None:
                raise RuntimeError(f"{self.name} is not running")
            self.proc.stdin.write(text)
            self.proc.stdin.flush()

    def stop(self) -> None:
        with self.lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return
            self.phase = "stopping"
            self.ready = False
            self.last_update_at = time.time()
            emit("status", f"正在停止 {self.name} pid={proc.pid}")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()

    def status(self) -> dict[str, Any]:
        proc = self.proc
        return {
            "name": self.name,
            "running": self.is_running(),
            "pid": None if proc is None else proc.pid,
            "returncode": None if proc is None else proc.poll(),
            "cmd": self.cmd,
            "started_at": self.started_at,
            "phase": self.phase,
            "ready": self.ready,
            "last_result": self.last_result,
            "last_error": self.last_error,
            "last_json": self.last_json,
            "last_update_at": self.last_update_at,
            "raw_log_path": None if self.raw_log_path is None else str(self.raw_log_path),
        }


class ResidentProcess(ManagedProcess):
    def send_json(self, payload: dict) -> None:
        cmd = str(payload.get("cmd") or "")
        with self.lock:
            if cmd == "warmup":
                self.phase = "warming"
                self.ready = False
                self.last_error = None
                self.last_update_at = time.time()
            elif cmd == "shutdown":
                self.phase = "stopping"
                self.ready = False
                self.last_update_at = time.time()
        self.send_stdin(json.dumps(payload, ensure_ascii=False) + "\n")

    def request_json(self, payload: dict, timeout: float = 300.0) -> dict:
        request_id = str(uuid.uuid4())
        payload = dict(payload)
        payload["request_id"] = request_id
        q: queue.Queue = queue.Queue(maxsize=1)
        with self.lock:
            self.pending[request_id] = q
        try:
            self.send_json(payload)
            response = q.get(timeout=float(timeout))
        except Exception:
            with self.lock:
                self.pending.pop(request_id, None)
            raise
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error") or response))
        return response


grasp_process = ManagedProcess("抓取流程")
perception_process = ManagedProcess("分割定位")
sam3_worker = ResidentProcess("SAM3 常驻")
sam6d_worker = ResidentProcess("SAM6D 常驻")
perception_task_lock = threading.Lock()
perception_task: dict[str, Any] = {
    "name": "分割定位",
    "phase": "idle",
    "running": False,
    "ready": False,
    "pid": None,
    "returncode": None,
    "started_at": None,
    "last_update_at": None,
    "last_result": None,
    "last_error": None,
    "last_json": None,
    "cmd": [],
    "object_names": [],
    "mask_found_objects": [],
    "mask_missing_objects": [],
    "pose_found_objects": [],
    "pose_missing_objects": [],
}


def _resident_models_active() -> bool:
    return sam3_worker.is_running() or sam6d_worker.is_running()


def _is_sam6d_provider_command(cmd: list[str]) -> bool:
    return any("sam6d_groundingdino_pose_provider.py" in str(item) for item in cmd)


def _is_sam6d_grasp_command(cmd: list[str]) -> bool:
    return any("rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place_sam6d.py" in str(item) for item in cmd)


def _command_has_fixed_scene_input(cmd: list[str]) -> bool:
    fixed_flags = {
        "--sam6d-fixed-scene-result-file",
        "--fixed-scene-pose-file",
        "--fixed-scene-result-file",
    }
    return any(str(item) in fixed_flags for item in cmd)


def _latest_perception_result_path() -> str | None:
    result_path = latest_perception_result.get("result_path")
    if not result_path:
        return None
    path = Path(str(result_path)).expanduser()
    if not path.exists():
        return None
    return str(path)


def _guard_or_rewrite_live_sam6d_command(cmd: list[str]) -> tuple[list[str], str | None]:
    if _is_sam6d_provider_command(cmd) and _resident_models_active():
        raise RuntimeError(
            "这个命令会启动独立 SAM6D/SAM3 provider，但常驻 SAM3/SAM6D 已占用显存；"
            "请用“重新分割定位”按钮走常驻接口，或先停止热启动。"
        )
    if not _is_sam6d_grasp_command(cmd) or _command_has_fixed_scene_input(cmd):
        return cmd, None
    fixed_result = _latest_perception_result_path()
    if fixed_result:
        rewritten = list(cmd) + ["--sam6d-fixed-scene-result-file", fixed_result]
        return rewritten, f"命令框未指定定位结果，已自动复用最近一次 SAM6D 结果：{fixed_result}"
    if _resident_models_active():
        raise RuntimeError(
            "抓取命令未指定 --sam6d-fixed-scene-result-file，会在抓取脚本内部再启动独立 SAM3/SAM6D，"
            "当前常驻模型已占用显存，容易爆显存。请先点“重新分割定位”，再启动抓取。"
        )
    return cmd, None


def _release_sam6d_for_fixed_scene_grasp(cmd: list[str]) -> bool:
    if not (_is_sam6d_grasp_command(cmd) and _command_has_fixed_scene_input(cmd)):
        return False
    if not sam6d_worker.is_running():
        return False
    emit("summary", "抓取将复用固定 SAM6D 定位结果，先释放 SAM6D 常驻显存，保留 SAM3 常驻。")
    proc = sam6d_worker.proc
    try:
        sam6d_worker.send_json({"cmd": "shutdown"})
    except Exception:
        pass
    deadline = time.time() + 1.5
    while proc is not None and proc.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    if proc is not None and proc.poll() is None:
        sam6d_worker.stop()
        deadline = time.time() + 6.0
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
    if proc is not None and proc.poll() is None:
        emit("summary", "SAM6D 常驻未及时退出，强制释放进程。")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass
    return True


def update_perception_task(**fields) -> dict[str, Any]:
    with perception_task_lock:
        perception_task.update(fields)
        perception_task["last_update_at"] = time.time()
        return dict(perception_task)


def current_perception_status() -> dict[str, Any]:
    with perception_task_lock:
        task = dict(perception_task)
    if task.get("running") or task.get("started_at") is not None:
        return task
    return perception_process.status()


def _sam3_result_object_marks(object_names: list[str], sam3_result: dict[str, Any]) -> tuple[list[str], list[str]]:
    requested = [str(name) for name in object_names]
    requested_set = set(requested)
    found: set[str] = set()
    for item in list(sam3_result.get("results") or []):
        if not isinstance(item, dict) or item.get("ok") is False:
            continue
        name = str(item.get("object_name") or "")
        if not name:
            raw_id = str(item.get("id") or "")
            name = raw_id.split("_p", 1)[0] if "_p" in raw_id else raw_id
        if name in requested_set:
            found.add(name)
    missing = [name for name in requested if name not in found]
    return [name for name in requested if name in found], missing


def _bbox_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in list(a)[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in list(b)[:4]]
    except Exception:
        return 0.0
    aw = max(0.0, ax2 - ax1)
    ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 1e-6 else float(inter / union)


def _item_bbox(item: dict[str, Any]):
    bbox = item.get("mask_bbox")
    if bbox is not None:
        return bbox
    selected = item.get("selected") if isinstance(item.get("selected"), dict) else {}
    return selected.get("box")


def _best_sam3_item_by_object(sam3_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for item in list(sam3_result.get("results") or []):
        if not isinstance(item, dict) or item.get("ok") is False:
            continue
        name = str(item.get("object_name") or "")
        if not name:
            raw_id = str(item.get("id") or "")
            name = raw_id.split("_p", 1)[0] if "_p" in raw_id else raw_id
        if not name:
            continue
        selected = item.get("selected") if isinstance(item.get("selected"), dict) else {}
        score = float(selected.get("model_score", item.get("score", -1.0)) or -1.0)
        old = best.get(name)
        old_selected = old.get("selected") if isinstance(old, dict) and isinstance(old.get("selected"), dict) else {}
        old_score = float(old_selected.get("model_score", old.get("score", -1.0) if old else -1.0) or -1.0) if old else -1.0
        if old is None or score > old_score:
            best[name] = item
    return best


def _sam3_duplicate_mask_conflicts(object_names: list[str], sam3_result: dict[str, Any]) -> list[dict[str, Any]]:
    requested = set(str(name) for name in object_names)
    best = {name: item for name, item in _best_sam3_item_by_object(sam3_result).items() if name in requested}
    names = sorted(best)
    conflicts: list[dict[str, Any]] = []
    for idx, a in enumerate(names):
        item_a = best[a]
        bbox_a = _item_bbox(item_a)
        pixels_a = int(item_a.get("mask_pixels") or 0)
        for b in names[idx + 1 :]:
            item_b = best[b]
            bbox_b = _item_bbox(item_b)
            pixels_b = int(item_b.get("mask_pixels") or 0)
            iou = _bbox_iou(bbox_a, bbox_b)
            if iou < 0.86:
                continue
            area_ratio = min(pixels_a, pixels_b) / max(pixels_a, pixels_b, 1)
            if area_ratio < 0.55:
                continue
            conflicts.append(
                {
                    "objects": [a, b],
                    "bbox_iou": round(float(iou), 4),
                    "mask_pixels": [pixels_a, pixels_b],
                    "mask_bbox": [bbox_a, bbox_b],
                    "scores": [
                        (item_a.get("selected") or {}).get("model_score"),
                        (item_b.get("selected") or {}).get("model_score"),
                    ],
                }
            )
    return conflicts


def _pose_result_object_marks(object_names: list[str], pose_result: dict[str, Any]) -> tuple[list[str], list[str]]:
    requested = [str(name) for name in object_names]
    requested_set = set(requested)
    found: set[str] = set()
    for item in list(pose_result.get("results") or []):
        if not isinstance(item, dict) or not item.get("ok"):
            continue
        name = str(item.get("object_name") or "")
        if name in requested_set:
            found.add(name)
    missing = [name for name in requested if name not in found]
    return [name for name in requested if name in found], missing


def _pose_duplicate_conflicts(object_names: list[str], pose_result: dict[str, Any], *, threshold_m: float = 0.018) -> list[dict[str, Any]]:
    requested = set(str(name) for name in object_names)
    items = []
    for item in list(pose_result.get("results") or []):
        if not isinstance(item, dict) or not item.get("ok"):
            continue
        name = str(item.get("object_name") or "")
        if name not in requested:
            continue
        t = item.get("translation_m")
        if t is None and isinstance(item.get("T_cam_obj"), list):
            mat = item.get("T_cam_obj")
            try:
                t = [mat[0][3], mat[1][3], mat[2][3]]
            except Exception:
                t = None
        if t is None:
            continue
        try:
            trans = [float(v) for v in list(t)[:3]]
        except Exception:
            continue
        items.append((name, trans, item))
    conflicts = []
    for idx, (a, ta, item_a) in enumerate(items):
        for b, tb, item_b in items[idx + 1 :]:
            dist = sum((ta[i] - tb[i]) ** 2 for i in range(3)) ** 0.5
            if dist <= float(threshold_m):
                conflicts.append(
                    {
                        "objects": [a, b],
                        "distance_m": round(float(dist), 5),
                        "translation_m": [ta, tb],
                        "scores": [item_a.get("score"), item_b.get("score")],
                    }
                )
    return conflicts


def _pose_detail_map(pose_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    for item in list(pose_result.get("results") or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("object_name") or "")
        if not name:
            continue
        refine = item.get("pem_refine") if isinstance(item.get("pem_refine"), dict) else {}
        details[name] = {
            "ok": bool(item.get("ok")),
            "score": item.get("score"),
            "translation_m": item.get("translation_m"),
            "refine_applied": refine.get("applied") if refine else None,
            "center_error_px": (refine.get("refined_metrics") or {}).get("center_error_px") if isinstance(refine.get("refined_metrics"), dict) else None,
            "run_dir": item.get("run_dir"),
        }
    return details


def _latest_profile_path() -> Path | None:
    patterns = [
        str(ROOT / "planning_profile.jsonl"),
        str(ROOT / "planning_profile_logs" / "*.jsonl"),
        str(PICK_DIR / "planning_profile.jsonl"),
        str(PICK_DIR / "planning_profile_logs" / "*.jsonl"),
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(path) for path in glob.glob(pattern))
    existing = []
    for path in candidates:
        try:
            if path.exists():
                existing.append(path)
        except OSError:
            pass
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def latest_profile_waterfall(limit: int = 12) -> dict[str, Any]:
    path = _latest_profile_path()
    if path is None:
        return {"path": None, "stages": [], "objects": [], "total_ms": 0.0}
    stages: dict[str, dict[str, Any]] = {}
    objects: dict[str, dict[str, Any]] = {}
    rows = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                rows += 1
                stage = str(row.get("stage_name") or row.get("stage") or "unknown")
                obj = str(row.get("object_name") or row.get("object") or "")
                elapsed = row.get("elapsed_ms", row.get("total_ms", row.get("duration_ms", 0.0)))
                try:
                    elapsed_ms = float(elapsed or 0.0)
                except Exception:
                    elapsed_ms = 0.0
                ok_raw = row.get("ok", row.get("success", None))
                ok = bool(ok_raw) if ok_raw is not None else True
                item = stages.setdefault(stage, {"stage": stage, "total_ms": 0.0, "n": 0, "ok": 0, "max_ms": 0.0})
                item["total_ms"] += elapsed_ms
                item["n"] += 1
                item["ok"] += 1 if ok else 0
                item["max_ms"] = max(float(item["max_ms"]), elapsed_ms)
                if obj:
                    obj_item = objects.setdefault(obj, {"object": obj, "total_ms": 0.0, "n": 0, "ok": 0})
                    obj_item["total_ms"] += elapsed_ms
                    obj_item["n"] += 1
                    obj_item["ok"] += 1 if ok else 0
    except Exception as exc:
        return {"path": str(path), "error": repr(exc), "stages": [], "objects": [], "total_ms": 0.0}
    stage_list = sorted(stages.values(), key=lambda item: float(item["total_ms"]), reverse=True)
    object_list = sorted(objects.values(), key=lambda item: float(item["total_ms"]), reverse=True)
    return {
        "path": str(path),
        "mtime": path.stat().st_mtime,
        "rows": rows,
        "stages": stage_list[:limit],
        "objects": object_list[:8],
        "total_ms": float(sum(float(item["total_ms"]) for item in stages.values())),
    }


def _is_failure_log_item(item: dict[str, Any]) -> bool:
    text = str(item.get("message") or "").strip()
    if not text:
        return False
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            if payload.get("ok") is True:
                return False
            if payload.get("ok") is False:
                return True
    kind = str(item.get("kind") or "")
    if kind == "status" and "已退出 code=" in text and "code=0" not in text:
        return True
    hard_tokens = ("[FAIL]", "失败", "RuntimeError", "Traceback", "CUDA error", "out of memory")
    if any(token in text for token in hard_tokens):
        return True
    lower = text.lower()
    soft_tokens = ("failed:", "failed with code", "process exited with code", "collision")
    return any(token in lower for token in soft_tokens)


def latest_failure_summary() -> dict[str, Any]:
    messages = []
    for item in reversed(list(log_history)):
        text = str(item.get("message") or "")
        raw_kind = str(item.get("kind") or "")
        if _is_failure_log_item(item):
            messages.append(
                {
                    "ts": item.get("ts"),
                    "kind": raw_kind,
                    "process": item.get("process"),
                    "message": text[-800:],
                }
            )
        if len(messages) >= 5:
            break
    images = latest_files(["pick_jiaobang/failure_renders/*.png"], 3, min_mtime=SERVER_STARTED_AT)
    return {
        "has_failure": bool(messages or images),
        "messages": messages,
        "images": images,
        "latest_image": images[0] if images else None,
    }


def build_mapping_preview(config: dict[str, Any]) -> list[dict[str, Any]]:
    objects = _as_names(config.get("objects"), DEFAULT_GRASP_OBJECTS, [])
    slot_order = _slot_order_tokens(config.get("slot_order")) or ["1", "2", "3", "4", "5", "6"]
    source_map = {}
    for token in _source_slot_map_tokens(config.get("source_slot_map")):
        source, slot = token.split(":", 1)
        source_map[source] = slot
    preview = []
    slot_idx = 0
    for idx, name in enumerate(objects, start=1):
        if name == "bi":
            preview.append({"index": idx, "object": name, "destination": "bitong", "fixed": True})
            continue
        slot = source_map.get(name)
        if slot is None:
            slot = slot_order[slot_idx] if slot_idx < len(slot_order) else None
        slot_idx += 1
        preview.append({"index": idx, "object": name, "destination": None if slot is None else f"slot_{slot}", "fixed": False})
    return preview


def preflight_report(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config or {})
    checks: list[dict[str, Any]] = []
    objects = _as_names(config.get("objects"), DEFAULT_GRASP_OBJECTS, [])
    tracked = _as_names(config.get("tracked_objects"), DEFAULT_TRACKED_OBJECTS, DEFAULT_TRACKED_OBJECTS)
    reuse = True
    execute_real = bool(config.get("execute_real", False))
    source_slot_map = _source_slot_map_tokens(config.get("source_slot_map"))
    slot_sources = [token.split(":", 1)[0] for token in source_slot_map]

    def add(level: str, title: str, detail: str = "") -> None:
        checks.append({"level": level, "title": title, "detail": detail})

    add("ok" if sam3_worker.status().get("ready") else "warn", "SAM3 常驻", "ready" if sam3_worker.status().get("ready") else "尚未热启动，启动感知时会自动加载")
    add("ok" if sam6d_worker.status().get("ready") else "warn", "SAM6D 常驻", "ready" if sam6d_worker.status().get("ready") else "尚未热启动，启动感知时会自动加载")
    if not objects:
        add("bad", "抓取目标", "至少选择一个可抓目标")
    else:
        add("ok", "抓取目标", "，".join(objects))

    if reuse:
        result_path = latest_perception_result.get("result_path")
        pose_found = set(str(x) for x in latest_perception_result.get("pose_found_objects") or [])
        if not result_path or not Path(str(result_path)).exists():
            add("bad", "复用定位", "没有可复用的 SAM6D 结果，请先重新分割定位")
        else:
            missing = [name for name in objects if name not in pose_found]
            if missing:
                add("bad", "目标定位", f"这些目标没有成功定位：{missing}")
            else:
                age_s = time.time() - float(latest_perception_result.get("updated_at") or time.time())
                add("ok", "目标定位", f"最近定位 {age_s:.0f}s 前，pose {len(pose_found)} 个对象")
            required_scene = ["desk"]
            if "bi" in objects:
                required_scene.append("bitong")
            missing_scene = [name for name in required_scene if name not in pose_found]
            if missing_scene:
                add("bad", "放置参照物", f"缺少定位：{missing_scene}")
            else:
                add("ok", "放置参照物", "desk/bitong 状态满足当前目标")
    tabletop = [name for name in objects if name != "bi"]
    if len(tabletop) > 6:
        add("bad", "slot 数量", f"桌面目标 {len(tabletop)} 个，超过 6 个 slot")
    elif len(set(slot_sources)) != len(slot_sources):
        add("bad", "slot 映射", "存在重复映射源")
    else:
        add("ok", "slot 映射", "；".join(source_slot_map) if source_slot_map else "按右侧 slot 顺序自动分配")

    if "desk" not in tracked:
        add("warn", "tracked 对象", "desk 不在 tracked 列表中")
    if "bi" in objects and "bitong" not in tracked:
        add("bad", "笔筒绑定", "bi 需要 bitong 被跟踪/定位")
    if execute_real:
        add("warn", "真机执行", f"real_control_hz={int(config.get('real_control_hz') or 30)} max_delta={float(config.get('real_max_delta_per_step') or 0.1):.3f}")
    else:
        add("ok", "执行模式", "不会发送真机动作")

    ok = not any(item["level"] == "bad" for item in checks)
    return {
        "ok": ok,
        "checks": checks,
        "mapping": build_mapping_preview(config),
        "mode": "real" if execute_real else "dry",
    }


def generate_placement_preview(config: dict[str, Any]) -> dict[str, Any]:
    import sam6d_groundingdino_pose_provider as provider

    config = dict(config or {})
    result_path = config.get("sam6d_fixed_scene_result_file") or latest_perception_result.get("result_path")
    if not result_path:
        raise RuntimeError("没有可用的 SAM6D 定位结果，请先重新分割定位")
    result_path = Path(str(result_path)).expanduser().resolve()
    if not result_path.exists():
        raise FileNotFoundError(str(result_path))
    summary = json.loads(result_path.read_text(encoding="utf-8"))
    scene_dir = Path(str(summary.get("scene_dir") or result_path.parent)).expanduser()
    frame_dir = scene_dir / "shared_frame"
    rgb_path = frame_dir / "rgb.png"
    depth_path = frame_dir / "depth.png"
    camera_path = frame_dir / "camera.json"
    missing = [str(path) for path in (rgb_path, depth_path, camera_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"放置预览缺少相机帧文件: {missing}")

    args = argparse.Namespace(
        rgb_path=str(rgb_path),
        depth_path=str(depth_path),
        camera_path=str(camera_path),
        camera_extrinsic_opencv_path=DEFAULT_CAMERA_EXTRINSIC_OPENCV,
        use_direct_camera_extrinsic=False,
        object_name=None,
    )
    frame = provider.load_offline_frame(args)
    assignments = build_mapping_preview(config)
    out_path = scene_dir / "placement_preview.png"
    info = provider.save_placement_preview_visualization(
        args,
        frame,
        list(summary.get("results") or []),
        assignments,
        out_path,
    )
    return {
        "path": str(out_path.resolve()),
        "rel": str(out_path.resolve().relative_to(ROOT.resolve())) if str(out_path.resolve()).startswith(str(ROOT.resolve())) else str(out_path),
        "assignments": assignments,
        "info": info,
    }


@app.post("/api/placement-preview")
def api_placement_preview():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        config = dict(payload.get("config") or payload)
        result = generate_placement_preview(config)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, **result})


def run_gpu_sampler(stop_event: threading.Event) -> None:
    query = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    while not stop_event.is_set():
        sample = {"ts": now_ms(), "ok": False}
        try:
            out = subprocess.check_output(query, text=True, timeout=2.0).strip().splitlines()
            if out:
                parts = [item.strip() for item in out[0].split(",")]
                util, mem_used, mem_total, temp, power = parts[:5]
                sample.update(
                    {
                        "ok": True,
                        "util": float(util),
                        "mem_used": float(mem_used),
                        "mem_total": float(mem_total),
                        "temp": float(temp),
                        "power": float(power),
                    }
                )
        except Exception as exc:
            sample["error"] = repr(exc)
        gpu_history.append(sample)
        emit("gpu", **sample)
        stop_event.wait(1.0)


gpu_stop = threading.Event()
threading.Thread(target=run_gpu_sampler, args=(gpu_stop,), daemon=True).start()


def latest_files(patterns: list[str], limit: int = 8, *, min_mtime: float | None = None) -> list[dict[str, Any]]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(ROOT.glob(pattern))
    out = []
    for path in files:
        try:
            st = path.stat()
        except OSError:
            continue
        if min_mtime is not None and st.st_mtime < float(min_mtime):
            continue
        out.append({"path": str(path.resolve()), "name": path.name, "mtime": st.st_mtime, "rel": str(path.relative_to(ROOT))})
    out.sort(key=lambda item: item["mtime"], reverse=True)
    return out[:limit]


def safe_image_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    allowed_roots = [ROOT.resolve(), Path("/tmp").resolve()]
    if not any(str(path).startswith(str(root)) for root in allowed_roots):
        raise ValueError(f"image path outside allowed roots: {path}")
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


@app.get("/")
def index() -> str:
    return INDEX_HTML


@app.get("/events")
def events() -> Response:
    q: queue.Queue = queue.Queue(maxsize=200)
    with event_lock:
        event_clients.append(q)

    def gen():
        try:
            for item in list(log_history)[-100:]:
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            while True:
                try:
                    item = q.get(timeout=15.0)
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with event_lock:
                if q in event_clients:
                    event_clients.remove(q)

    return Response(gen(), mimetype="text/event-stream")


@app.get("/api/status")
def api_status():
    process = psutil.Process(os.getpid())
    return jsonify(
        {
            "server_pid": os.getpid(),
            "rss_mb": round(process.memory_info().rss / 1024 / 1024, 1),
            "sam3": sam3_worker.status(),
            "sam6d": sam6d_worker.status(),
            "perception": current_perception_status(),
            "grasp": grasp_process.status(),
            "latest_perception_result": dict(latest_perception_result),
            "profile": latest_profile_waterfall(),
            "failure": latest_failure_summary(),
            "run_id": time.strftime("%Y%m%d_%H%M%S", time.localtime(SERVER_STARTED_AT)),
            "server_started_at": SERVER_STARTED_AT,
            "gpu": list(gpu_history)[-120:],
            "logs": list(log_history)[-200:],
        }
    )


@app.post("/api/preflight")
def api_preflight():
    payload = request.get_json(force=True, silent=True) or {}
    config = dict(payload.get("config") or payload)
    try:
        report = preflight_report(config)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "report": report})


@app.post("/api/hotstart/sam3")
def api_hotstart_sam3():
    payload = request.get_json(force=True, silent=True) or {}
    status = ensure_sam3_resident(payload, wait=False)
    emit("summary", "已请求 SAM3 常驻热启动。")
    return jsonify({"ok": True, "status": status})


def ensure_sam3_resident(payload: dict | None = None, *, wait: bool = True) -> dict[str, Any]:
    payload = dict(payload or {})
    python_path = str(payload.get("python") or DEFAULT_SAM3_PYTHON)
    cmd = [
        python_path,
        str(PICK_DIR / "sam3_resident_worker.py"),
        "--checkpoint-path",
        str(payload.get("checkpoint_path") or DEFAULT_SAM3_CHECKPOINT),
        "--device",
        str(payload.get("device") or "cuda"),
        "--resolution",
        str(int(payload.get("resolution") or 1008)),
        "--confidence-threshold",
        str(float(payload.get("confidence_threshold") or 0.35)),
    ]
    if not sam3_worker.is_running():
        sam3_worker.start(cmd, cwd=PICK_DIR)
    if wait:
        if not bool(sam3_worker.status().get("ready", False)):
            sam3_worker.request_json({"cmd": "warmup"}, timeout=180.0)
    else:
        sam3_worker.send_json({"cmd": "warmup"})
    return sam3_worker.status()


@app.post("/api/hotstart/sam6d")
def api_hotstart_sam6d():
    payload = request.get_json(force=True, silent=True) or {}
    status = ensure_sam6d_resident(payload, wait=False)
    emit("summary", "已请求 SAM6D PEM 常驻热启动和模板特征预加载。")
    return jsonify({"ok": True, "status": status})


def ensure_sam6d_resident(payload: dict | None = None, *, wait: bool = True) -> dict[str, Any]:
    payload = dict(payload or {})
    cmd = [
        str(payload.get("python") or DEFAULT_FOUNDATIONPOSE_PYTHON),
        str(PICK_DIR / "sam6d_resident_worker.py"),
        "--sam6d-root",
        str(payload.get("sam6d_root") or "/home/zhangzhao/PycharmProjects/SAM-6D/SAM-6D"),
        "--template-cache-root",
        str(payload.get("template_cache_root") or PICK_DIR / "sam6d_template_cache"),
        "--pem-feature-cache-root",
        str(payload.get("pem_feature_cache_root") or PICK_DIR / "sam6d_pem_feature_cache"),
    ]
    if not sam6d_worker.is_running():
        sam6d_worker.start(cmd, cwd=PICK_DIR)
    warmup_payload = {"cmd": "warmup", "object_names": payload.get("object_names") or DEFAULT_OBJECTS}
    if wait:
        if not bool(sam6d_worker.status().get("ready", False)):
            sam6d_worker.request_json(warmup_payload, timeout=240.0)
    else:
        sam6d_worker.send_json(warmup_payload)
    return sam6d_worker.status()


@app.post("/api/hotstart/all")
def api_hotstart_all():
    api_hotstart_sam3()
    api_hotstart_sam6d()
    return jsonify({"ok": True})


@app.post("/api/hotstart/stop")
def api_hotstart_stop():
    try:
        if sam3_worker.is_running():
            sam3_worker.send_json({"cmd": "shutdown"})
    except Exception:
        pass
    try:
        if sam6d_worker.is_running():
            sam6d_worker.send_json({"cmd": "shutdown"})
    except Exception:
        pass
    sam3_worker.stop()
    sam6d_worker.stop()
    return jsonify({"ok": True})


@app.post("/api/perception/command")
def api_perception_command():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        cmd = build_perception_command(payload.get("config") or payload)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "command": shell_join(cmd), "argv": cmd})


def _resident_provider_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_root": str(PICK_DIR / "sam6d_grasp_scene_runs"),
        "foundationpose_root": "/home/zhangzhao/PycharmProjects/FoundationPose",
        "sam6d_root": "/home/zhangzhao/PycharmProjects/SAM-6D/SAM-6D",
        "template_cache_root": str(PICK_DIR / "sam6d_template_cache"),
        "pem_feature_cache_root": str(PICK_DIR / "sam6d_pem_feature_cache"),
        "mask_mode": "sam3_text",
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 30,
        "warmup_frames": 30,
        "camera_extrinsic_opencv_path": DEFAULT_CAMERA_EXTRINSIC_OPENCV,
        "use_direct_camera_extrinsic": False,
        "sam3_morph_kernel": 3,
        "sam3_max_masks_per_item": 1,
        "min_mask_area": 64,
        "sam3_require_full_scene_masks": bool(config.get("confirm_segmentation", True)),
        "sam3_full_scene_mask_confirm": False,
        "sam3_show_full_scene_mask_window": False,
        "full_scene_pem_visualization": True,
        "pem_save_visualization": True,
        "pem_run_mode": "inprocess",
        "post_pem_mask_refine": False,
        "post_pem_mask_refine_objects": "lvmukuai,carriot,tennis",
        "post_pem_mask_refine_trigger_px": 6.0,
    }


def _run_resident_perception(config: dict[str, Any]) -> None:
    global latest_perception_result
    object_names = _as_names(config.get("object_names"), DEFAULT_OBJECTS, DEFAULT_OBJECTS)
    started_at = time.time()
    update_perception_task(
        phase="warming",
        running=True,
        ready=False,
        returncode=None,
        started_at=started_at,
        last_result=None,
        last_error=None,
        last_json=None,
        cmd=["resident_sam3_sam6d"],
        object_names=object_names,
        mask_found_objects=[],
        mask_missing_objects=[],
        pose_found_objects=[],
        pose_missing_objects=[],
    )
    emit("summary", "分割定位：使用常驻 SAM3/SAM6D，开始检查热启动状态。")
    try:
        ensure_sam3_resident({}, wait=True)
        ensure_sam6d_resident({"object_names": object_names}, wait=True)

        provider_payload = _resident_provider_payload(config)
        update_perception_task(phase="capturing", running=True)
        emit("summary", "分割定位：SAM6D 常驻正在拍照。")
        capture_resp = sam6d_worker.request_json(
            {"cmd": "capture_frame", "object_names": object_names, **provider_payload},
            timeout=120.0,
        )
        capture = capture_resp.get("result") or {}
        emit(
            "summary",
            f"分割定位：拍照完成，RealSense 已释放。frame={capture.get('frame_dir', '')}",
        )

        update_perception_task(phase="segmenting", running=True, last_result=capture)
        emit("summary", "分割定位：SAM3 常驻正在分割。")
        sam3_resp = sam3_worker.request_json(
            {
                "cmd": "segment",
                "rgb_path": capture["rgb_path"],
                "output_dir": str(Path(capture["scene_dir"]) / "sam3_full_scene_text"),
                "items": capture["items"],
                "morph_kernel": int(provider_payload["sam3_morph_kernel"]),
                "min_mask_area": int(provider_payload["min_mask_area"]),
            },
            timeout=240.0,
        )
        sam3_result = sam3_resp.get("result") or {}
        mask_found, mask_missing = _sam3_result_object_marks(object_names, sam3_result)
        sam3_result["mask_found_objects"] = mask_found
        sam3_result["mask_missing_objects"] = mask_missing
        mask_conflicts = _sam3_duplicate_mask_conflicts(object_names, sam3_result)
        sam3_result["mask_conflicts"] = mask_conflicts
        if mask_conflicts:
            update_perception_task(
                phase="error",
                running=False,
                ready=False,
                returncode=1,
                last_result=sam3_result,
                mask_found_objects=mask_found,
                mask_missing_objects=mask_missing,
                pose_found_objects=[],
                pose_missing_objects=list(object_names),
            )
            raise RuntimeError(f"SAM3 mask matched multiple object names; conflicts={mask_conflicts}")
        pose_object_names = [name for name in object_names if name in set(mask_found)]
        if not pose_object_names:
            update_perception_task(
                phase="error",
                running=False,
                ready=False,
                returncode=1,
                last_result=sam3_result,
                mask_found_objects=mask_found,
                mask_missing_objects=mask_missing,
                pose_found_objects=[],
                pose_missing_objects=list(object_names),
            )
            raise RuntimeError(f"SAM3 did not segment any requested object; missing={mask_missing}")

        update_perception_task(
            phase="posing",
            running=True,
            last_result=sam3_result,
            mask_found_objects=mask_found,
            mask_missing_objects=mask_missing,
        )
        emit(
            "summary",
            f"分割定位：SAM3 找到 {len(mask_found)}/{len(object_names)}，"
            f"缺失={mask_missing or []}；只对已分割对象做 SAM6D 定位。",
        )
        pose_resp = sam6d_worker.request_json(
            {
                "cmd": "pose_from_sam3",
                "object_names": pose_object_names,
                "frame_dir": capture["frame_dir"],
                "scene_dir": capture["scene_dir"],
                "sam3_result_path": sam3_result["result_path"],
                **provider_payload,
            },
            timeout=360.0,
        )
        result = pose_resp.get("result") or {}
        ok_count = int(result.get("ok_count", 0) or 0)
        pose_object_count = int(result.get("object_count", len(pose_object_names)) or len(pose_object_names))
        pose_found, pose_missing = _pose_result_object_marks(object_names, result)
        pose_conflicts = _pose_duplicate_conflicts(object_names, result)
        result["pose_conflicts"] = pose_conflicts
        result["mask_found_objects"] = mask_found
        result["mask_missing_objects"] = mask_missing
        result["pose_found_objects"] = pose_found
        result["pose_missing_objects"] = pose_missing
        result["requested_object_names"] = object_names
        result["pose_object_names"] = pose_object_names
        result["requested_object_count"] = len(object_names)
        result["pose_object_count"] = pose_object_count
        result["pose_details"] = _pose_detail_map(result)
        if pose_conflicts:
            update_perception_task(
                phase="error",
                running=False,
                ready=False,
                returncode=1,
                last_result=result,
                last_json=pose_resp,
                mask_found_objects=mask_found,
                mask_missing_objects=mask_missing,
                pose_found_objects=pose_found,
                pose_missing_objects=pose_missing,
            )
            raise RuntimeError(f"SAM6D produced overlapping object poses; conflicts={pose_conflicts}")
        latest_perception_result = {
            "result_path": result.get("result_path"),
            "scene_dir": result.get("scene_dir"),
            "object_names": object_names,
            "ok_count": ok_count,
            "object_count": len(object_names),
            "pose_object_count": pose_object_count,
            "mask_found_objects": mask_found,
            "mask_missing_objects": mask_missing,
            "pose_found_objects": pose_found,
            "pose_missing_objects": pose_missing,
            "pose_details": result["pose_details"],
            "updated_at": time.time(),
        }
        update_perception_task(
            phase="stopped",
            running=False,
            ready=True,
            returncode=0,
            last_result=result,
            last_error=None,
            last_json=pose_resp,
            mask_found_objects=mask_found,
            mask_missing_objects=mask_missing,
            pose_found_objects=pose_found,
            pose_missing_objects=pose_missing,
        )
        emit(
            "summary",
            f"分割定位完成：SAM3 mask={len(mask_found)}/{len(object_names)}，"
            f"SAM6D pose={ok_count}/{pose_object_count} result={result.get('result_path')}",
        )
    except Exception as exc:
        update_perception_task(
            phase="error",
            running=False,
            ready=False,
            returncode=1,
            last_error=repr(exc),
        )
        emit("summary", f"分割定位失败：{exc!r}")


@app.post("/api/perception/run")
def api_perception_run():
    payload = request.get_json(force=True, silent=True) or {}
    command = str(payload.get("command") or "").strip()
    if not command:
        config = payload.get("config") or payload
        object_names = _as_names(config.get("object_names"), DEFAULT_OBJECTS, DEFAULT_OBJECTS)
        if not object_names:
            return jsonify({"ok": False, "error": "至少选择一个分割定位对象"}), 400
        current = current_perception_status()
        if current.get("running"):
            return jsonify({"ok": False, "error": "分割定位正在运行"}), 409
        threading.Thread(target=_run_resident_perception, args=(dict(config),), daemon=True).start()
        return jsonify({"ok": True, "status": current_perception_status(), "resident": True})

    try:
        cmd = shlex.split(command) if command else build_perception_command(payload.get("config") or payload)
        cmd, rewrite_note = _guard_or_rewrite_live_sam6d_command(cmd)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    if rewrite_note:
        emit("summary", rewrite_note)
    perception_process.start(cmd, cwd=ROOT)
    emit("summary", "已启动一次分割定位流程；这不会启动机械臂抓取。")
    return jsonify({"ok": True, "status": perception_process.status(), "command": shell_join(cmd)})


@app.post("/api/perception/stdin")
def api_perception_stdin():
    payload = request.get_json(force=True, silent=True) or {}
    text = str(payload.get("text") or "")
    perception_process.send_stdin(text)
    return jsonify({"ok": True})


@app.post("/api/perception/stop")
def api_perception_stop():
    perception_process.stop()
    update_perception_task(phase="stopped", running=False, returncode=0, ready=False)
    return jsonify({"ok": True})


@app.post("/api/grasp/command")
def api_grasp_command():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        config = dict(payload.get("config") or payload)
        config["reuse_latest_perception"] = True
        if config.get("reuse_latest_perception") and not config.get("sam6d_fixed_scene_result_file") and not latest_perception_result.get("result_path"):
            return jsonify({"ok": False, "error": "还没有可复用的分割定位结果，请先点“重新分割定位”"}), 400
        cmd = build_grasp_command(config)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "command": shell_join(cmd), "argv": cmd})


@app.post("/api/grasp/start")
def api_grasp_start():
    payload = request.get_json(force=True, silent=True) or {}
    if payload.get("config") is not None:
        try:
            config = dict(payload.get("config") or {})
            config["reuse_latest_perception"] = True
            if config.get("reuse_latest_perception") and not config.get("sam6d_fixed_scene_result_file") and not latest_perception_result.get("result_path"):
                return jsonify({"ok": False, "error": "还没有可复用的分割定位结果，请先点“重新分割定位”"}), 400
            report = preflight_report(config)
            if not report.get("ok"):
                bad = [item for item in report.get("checks", []) if item.get("level") == "bad"]
                return jsonify({"ok": False, "error": "运行前检查未通过: " + "; ".join(str(item.get("title")) + " " + str(item.get("detail") or "") for item in bad), "preflight": report}), 400
            cmd = build_grasp_command(config)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        command = shell_join(cmd)
        emit("summary", f"抓取顺序/放置映射：{placement_mapping_text(config)}")
    else:
        command = str(payload.get("command") or DEFAULT_GRASP_COMMAND).strip()
        try:
            cmd = shlex.split(command) if command else []
            cmd, rewrite_note = _guard_or_rewrite_live_sam6d_command(cmd)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if rewrite_note:
            emit("summary", rewrite_note)
            command = shell_join(cmd)
    if not command:
        return jsonify({"ok": False, "error": "empty command"}), 400
    _release_sam6d_for_fixed_scene_grasp(cmd)
    grasp_process.start(cmd, cwd=ROOT)
    emit("summary", "抓取流程已从 Web 启动。")
    return jsonify({"ok": True, "status": grasp_process.status(), "command": command})


@app.post("/api/grasp/stdin")
def api_grasp_stdin():
    payload = request.get_json(force=True, silent=True) or {}
    text = str(payload.get("text") or "")
    grasp_process.send_stdin(text)
    return jsonify({"ok": True})


@app.post("/api/grasp/stop")
def api_grasp_stop():
    grasp_process.stop()
    return jsonify({"ok": True})


def latest_image_payload() -> dict[str, list[dict[str, Any]]]:
    return {
        "sam3": latest_files(["pick_jiaobang/sam6d_grasp_scene_runs/*/sam3_full_scene_text/sam3_full_scene_masks_overlay.png"], 5, min_mtime=SERVER_STARTED_AT),
        "pem": latest_files(["pick_jiaobang/sam6d_grasp_scene_runs/*/full_scene_pem_overlay.png"], 5, min_mtime=SERVER_STARTED_AT),
        "placement": latest_files(["pick_jiaobang/sam6d_grasp_scene_runs/*/placement_preview.png"], 5, min_mtime=SERVER_STARTED_AT),
        "sapien": latest_files(["pick_jiaobang/failure_renders/*.png"], 5, min_mtime=SERVER_STARTED_AT),
        "rgb": latest_files(["pick_jiaobang/sam6d_grasp_scene_runs/*/shared_frame/rgb.png"], 5, min_mtime=SERVER_STARTED_AT),
    }


@app.get("/api/latest-images")
def api_latest_images():
    return jsonify(latest_image_payload())


@app.post("/api/debug-pack")
def api_debug_pack():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PICK_DIR / "web_debug_packs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / f"{stamp}_web_debug_pack.tar.gz"
    manifest = {
        "created_at": time.time(),
        "server_started_at": SERVER_STARTED_AT,
        "latest_perception_result": dict(latest_perception_result),
        "profile": latest_profile_waterfall(limit=40),
        "failure": latest_failure_summary(),
        "logs": list(log_history)[-500:],
        "sam3": sam3_worker.status(),
        "sam6d": sam6d_worker.status(),
        "perception": current_perception_status(),
        "grasp": grasp_process.status(),
    }
    manifest_path = out_dir / f"{stamp}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths: list[Path] = [manifest_path]
    profile_path = _latest_profile_path()
    if profile_path is not None:
        paths.append(profile_path)
    result_path = latest_perception_result.get("result_path")
    if result_path:
        paths.append(Path(str(result_path)).expanduser())
    for group in latest_image_payload().values():
        for item in list(group or []):
            try:
                paths.append(Path(str(item.get("path"))))
            except Exception:
                pass
    added: set[str] = set()
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in paths:
            try:
                path = path.resolve()
                if not path.exists() or str(path) in added:
                    continue
                added.add(str(path))
                arcname = path.name if path == manifest_path else str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else path.name
                tar.add(path, arcname=arcname)
            except Exception:
                continue
    return jsonify({"ok": True, "path": str(tar_path), "file_count": len(added)})


@app.get("/image")
def image():
    path = safe_image_path(str(request.args.get("path") or ""))
    return send_file(path)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RM75 抓取控制台</title>
  <style>
    :root {
      --bg: #eef1f5;
      --panel: #ffffff;
      --line: #d8dde3;
      --text: #1f2933;
      --muted: #66717f;
      --green: #15803d;
      --red: #b42318;
      --amber: #a16207;
      --blue: #2563eb;
      --ink: #0f172a;
      --cyan: #0891b2;
      --violet: #6d5bd0;
      --shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }
    header {
      min-height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 18px;
      border-bottom: 1px solid var(--line);
      background: #101827;
      color: #f8fafc;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 700; }
    header .note { color: #cbd5e1; }
    main {
      display: grid;
      grid-template-columns: minmax(390px, 0.92fr) minmax(560px, 1.38fr);
      gap: 14px;
      padding: 14px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px;
      min-width: 0;
      box-shadow: var(--shadow);
    }
    h2 {
      margin: 0 0 10px;
      font-size: 15px;
      font-weight: 650;
    }
    .row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .stack { display: grid; gap: 12px; }
    button {
      height: 34px;
      padding: 0 12px;
      border: 1px solid #c5cbd3;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
      font-size: 14px;
    }
    button.primary { background: var(--blue); color: #fff; border-color: var(--blue); }
    button.danger { background: #fff5f5; color: var(--red); border-color: #f3b7b1; }
    button.ghost { background: #f8fafc; color: var(--ink); }
    button:disabled { opacity: 0.45; cursor: default; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 28px;
      padding: 4px 9px;
      border-radius: 999px;
      background: #e8eef8;
      color: #23415f;
      font-size: 12px;
      font-weight: 650;
    }
    .pill.ok { background: #dcfce7; color: #166534; }
    .pill.warn { background: #fef3c7; color: #92400e; }
    .pill.bad { background: #fee2e2; color: #991b1b; }
    .dashboard-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }
    .metric-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfe;
      min-height: 68px;
    }
    .metric-card b {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 5px;
    }
    .metric-card span {
      font-size: 16px;
      color: var(--ink);
      font-weight: 700;
    }
    .timeline {
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 6px;
      margin-top: 10px;
    }
    .timeline-step {
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px;
      background: #f8fafc;
      font-size: 12px;
      color: var(--muted);
    }
    .timeline-step.active { border-color: #93c5fd; background: #eff6ff; color: #1d4ed8; }
    .timeline-step.done { border-color: #86efac; background: #f0fdf4; color: #166534; }
    .timeline-step.bad { border-color: #fecaca; background: #fff1f2; color: #991b1b; }
    textarea {
      width: 100%;
      min-height: 164px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 8px;
    }
    .status {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fafafa;
    }
    .status b { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .status small { display: block; margin-top: 3px; color: var(--muted); line-height: 1.3; overflow-wrap: anywhere; }
    .ok { color: var(--green); }
    .bad { color: var(--red); }
    .warn { color: var(--amber); }
    .check-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(112px, 1fr));
      gap: 7px;
    }
    label.check {
      display: flex;
      align-items: center;
      gap: 7px;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 8px;
      background: #fafafa;
      font-size: 13px;
    }
    label.check input { margin: 0; }
    label.check.asset-found {
      border-color: #86efac;
      background: #f0fdf4;
    }
    label.check.asset-missing {
      border-color: #fecaca;
      background: #fff1f2;
    }
    label.check .asset-status {
      margin-left: auto;
      font-size: 12px;
      white-space: nowrap;
    }
    label.check.asset-found .asset-status { color: var(--green); }
    label.check.asset-missing .asset-status { color: var(--red); }
    .field-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
    }
    .field label {
      display: block;
      margin-bottom: 5px;
      font-size: 12px;
      color: var(--muted);
    }
    input[type="text"], input[type="number"], select {
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 9px;
      background: #fff;
      font-size: 14px;
    }
    .mapping-board {
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) minmax(110px, 0.9fr);
      gap: 10px;
      align-items: start;
    }
    .drag-list {
      display: grid;
      gap: 7px;
      min-height: 44px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px;
      background: #fafafa;
    }
    .drag-card {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
      background: #fff;
      cursor: grab;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-size: 13px;
      user-select: none;
    }
    .drag-card:active { cursor: grabbing; }
    .drag-card.dragging { opacity: 0.45; }
    .drag-card.fixed {
      background: #f3f4f6;
      cursor: default;
    }
    .drag-card .badge {
      flex: 0 0 auto;
      color: var(--muted);
      font-size: 12px;
    }
    .mapping-line {
      margin-top: 7px;
      min-height: 18px;
    }
    .preview-list, .target-grid, .preflight-list {
      display: grid;
      gap: 8px;
    }
    .target-grid {
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    }
    .target-card, .preflight-item, .sequence-item, .failure-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fbfcfe;
      min-width: 0;
    }
    .target-card b, .sequence-item b { display: block; font-size: 14px; margin-bottom: 5px; }
    .target-card small, .sequence-item small, .preflight-item small { color: var(--muted); overflow-wrap: anywhere; }
    .target-card.ok, .preflight-item.ok { border-color: #86efac; background: #f0fdf4; }
    .target-card.warn, .preflight-item.warn { border-color: #fde68a; background: #fffbeb; }
    .target-card.bad, .preflight-item.bad, .failure-card.bad { border-color: #fecaca; background: #fff1f2; }
    .sequence-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
    }
    #waterfallChart {
      width: 100%;
      height: 230px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fcfcfd;
    }
    .subhead {
      margin: 10px 0 7px;
      font-size: 13px;
      font-weight: 650;
      color: var(--ink);
    }
    #gpuChart {
      width: 100%;
      height: 190px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fcfcfd;
    }
    #log {
      height: 300px;
      overflow: auto;
      background: #111827;
      color: #d1fae5;
      border-radius: 6px;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .images {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .image-slot {
      border: 1px solid var(--line);
      border-radius: 6px;
      min-height: 220px;
      background: #f9fafb;
      overflow: hidden;
      position: relative;
    }
    .image-slot img { width: 100%; display: block; }
    .empty-image {
      min-height: 190px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 13px;
      background:
        linear-gradient(45deg, #f3f4f6 25%, transparent 25%),
        linear-gradient(-45deg, #f3f4f6 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #f3f4f6 75%),
        linear-gradient(-45deg, transparent 75%, #f3f4f6 75%);
      background-size: 18px 18px;
      background-position: 0 0, 0 9px, 9px -9px, -9px 0;
    }
    .caption {
      padding: 7px 9px;
      border-bottom: 1px solid var(--line);
      font-size: 12px;
      color: var(--muted);
      background: #fff;
      overflow-wrap: anywhere;
    }
    .note { font-size: 12px; color: var(--muted); line-height: 1.4; }
    @media (max-width: 1000px) {
      main { grid-template-columns: 1fr; }
      .images { grid-template-columns: 1fr; }
      .mapping-board { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>RM75 抓取控制台</h1>
      <div class="note" id="runMeta">run 初始化中</div>
    </div>
    <div class="row">
      <span id="serverStatus" class="note">连接中...</span>
      <span class="pill" id="modePill">未启动</span>
      <button id="refreshBtn">刷新</button>
    </div>
  </header>
  <main>
    <div class="stack">
      <section>
        <h2>运行总览</h2>
        <div class="dashboard-grid">
          <div class="metric-card"><b>执行模式</b><span id="safetyMode">-</span></div>
          <div class="metric-card"><b>真机参数</b><span id="realParamText">-</span></div>
          <div class="metric-card"><b>最近定位</b><span id="perceptionMetric">-</span></div>
          <div class="metric-card"><b>预检状态</b><span id="preflightMetric">未检查</span></div>
        </div>
        <div class="timeline" id="stageTimeline"></div>
      </section>

      <section>
        <h2>常驻热启动</h2>
        <div class="status-grid">
          <div class="status"><b>SAM3</b><span id="sam3State">未知</span></div>
          <div class="status"><b>SAM6D</b><span id="sam6dState">未知</span></div>
          <div class="status"><b>分割定位</b><span id="perceptionState">未知</span></div>
          <div class="status"><b>抓取流程</b><span id="graspState">未知</span></div>
        </div>
        <div class="row" style="margin-top: 10px;">
          <button class="primary" id="hotAllBtn">热启动 SAM3 + SAM6D</button>
          <button id="hotSam3Btn">只启动 SAM3</button>
          <button id="hotSam6dBtn">只启动 SAM6D</button>
          <button class="danger" id="stopHotBtn">停止常驻</button>
        </div>
        <p class="note">热启动只加载模型和模板缓存，不会拍照、分割或定位。摆好物体后，用下面的“重新分割定位”按钮单独跑感知。</p>
      </section>

      <section>
        <h2>任务配置</h2>
        <div class="subhead">抓取目标</div>
        <div class="check-grid" id="graspObjectChecks"></div>
        <div class="row" style="margin-top: 8px;">
          <button id="selectAllGraspBtn">全选抓取目标</button>
          <button id="clearGraspBtn">清空抓取目标</button>
          <label class="check"><input type="checkbox" id="randomTargets" /> 随机顺序</label>
        </div>
        <div class="field-grid" style="margin-top: 8px;">
          <div class="field">
            <label for="executionMode">执行模式</label>
            <select id="executionMode">
              <option value="dry">只规划预览</option>
              <option value="sim">仿真运动窗口</option>
              <option value="real" selected>真机执行</option>
            </select>
          </div>
          <div class="field">
            <label for="realHz">real_control_hz</label>
            <input id="realHz" type="number" min="1" max="100" value="30" />
          </div>
          <div class="field">
            <label for="realDelta">real_max_delta_per_step</label>
            <input id="realDelta" type="number" min="0.01" max="0.5" step="0.01" value="0.1" />
          </div>
        </div>

        <div class="subhead">放置映射</div>
        <div class="mapping-board">
          <div class="field">
            <label>目标顺序，拖动卡片调整抓取优先级</label>
            <div class="drag-list" id="targetCardList"></div>
          </div>
          <div class="field">
            <label>桌面 slot 顺序，拖动卡片调整对应关系</label>
            <div class="drag-list" id="slotCardList"></div>
          </div>
        </div>
        <p class="note mapping-line" id="placementMappingText"></p>
        <div class="row" style="margin-top: 8px;">
          <button id="placementPreviewBtn">更新放置预览</button>
        </div>
        <div class="field-grid" style="margin-top: 8px;">
          <div class="field">
            <label for="renderMode">渲染模式</label>
            <select id="renderMode">
              <option value="human" selected>human</option>
              <option value="rgb_array">rgb_array</option>
              <option value="none">none</option>
            </select>
          </div>
        </div>
        <p class="note" id="latestPerceptionText">最近定位：暂无。正式抓取默认复用这里的结果，不会重新分割定位。</p>

        <div class="subhead">分割定位对象</div>
        <div class="check-grid" id="perceptionObjectChecks"></div>
        <div class="row" style="margin-top: 8px;">
          <button class="primary" id="runPerceptionBtn">重新分割定位</button>
          <button id="syncPerceptionBtn">感知对象同步抓取选择</button>
          <button class="danger" id="stopPerceptionBtn">停止分割定位</button>
        </div>
        <p class="note">重新分割定位会复用热启动的 SAM3/SAM6D 常驻：SAM6D 拍照，SAM3 常驻分割，SAM6D 常驻做 PEM 位姿，不再额外启动一份 SAM3。</p>
      </section>

      <section>
        <h2>运行前检查</h2>
        <div class="preflight-list" id="preflightList"></div>
        <div class="subhead">抓取顺序预览</div>
        <div class="sequence-list" id="sequencePreview"></div>
        <div class="row" style="margin-top: 10px;">
          <button id="preflightBtn">重新检查</button>
          <button id="debugPackBtn">导出调试包</button>
        </div>
        <p class="note" id="debugPackText">调试包：暂无</p>
      </section>

      <section>
        <h2>抓取命令</h2>
        <textarea id="command"></textarea>
        <div class="row" style="margin-top: 10px;">
          <button id="buildCmdBtn">预览当前配置命令</button>
          <button class="primary" id="startConfiguredBtn">按配置开始抓取</button>
          <button id="startBtn">运行命令框（高级）</button>
          <button class="danger" id="stopBtn">停止抓取</button>
          <button id="enterBtn">发送 Enter</button>
          <button id="retryBtn">发送 r 重试</button>
          <button id="quitBtn">发送 q 退出</button>
        </div>
        <p class="note">“按配置开始抓取”会直接按当前界面配置生成并启动命令，不需要先点预览。命令框只用于调试；如果未指定定位结果，后端会自动复用最近一次 SAM6D 结果，避免额外启动一套感知模型。</p>
      </section>

      <section>
        <h2>显卡曲线</h2>
        <canvas id="gpuChart" width="700" height="220"></canvas>
        <p id="gpuText" class="note">等待采样...</p>
      </section>
    </div>

    <div class="stack">
      <section>
        <h2>目标状态</h2>
        <div class="target-grid" id="targetStatusGrid"></div>
      </section>

      <section>
        <h2>阶段耗时</h2>
        <canvas id="waterfallChart" width="760" height="260"></canvas>
        <p class="note" id="profileText">等待 profile...</p>
      </section>

      <section>
        <h2>失败归因</h2>
        <div class="failure-card" id="failureCard">暂无失败记录</div>
      </section>

      <section>
        <h2>关键日志</h2>
        <div id="log"></div>
      </section>

      <section>
        <h2>可视化</h2>
        <p class="note">SAPIEN 的 native human viewer 不能直接嵌入浏览器。这里先显示最新 SAM3/SAM6D overlay 和失败渲染截图；后续若要真流式 SAPIEN，需要在抓取进程里定期导出 human render camera 帧。</p>
        <div class="images">
          <div class="image-slot"><div class="caption" id="sam3Caption">SAM3 mask</div><div class="empty-image" id="sam3Empty">暂无</div><img id="sam3Img" /></div>
          <div class="image-slot"><div class="caption" id="pemCaption">SAM6D PEM</div><div class="empty-image" id="pemEmpty">暂无</div><img id="pemImg" /></div>
          <div class="image-slot"><div class="caption" id="placementCaption">放置预览</div><div class="empty-image" id="placementEmpty">暂无</div><img id="placementImg" /></div>
          <div class="image-slot"><div class="caption" id="rgbCaption">相机 RGB</div><div class="empty-image" id="rgbEmpty">暂无</div><img id="rgbImg" /></div>
          <div class="image-slot"><div class="caption" id="sapienCaption">SAPIEN/失败截图</div><div class="empty-image" id="sapienEmpty">暂无</div><img id="sapienImg" /></div>
        </div>
      </section>
    </div>
  </main>
<script>
const defaultCommand = __DEFAULT_COMMAND_JSON__;
const graspObjects = __GRASP_OBJECTS_JSON__;
const trackedObjects = __TRACKED_OBJECTS_JSON__;
const allSceneObjects = __ALL_OBJECTS_JSON__;
document.getElementById('command').value = defaultCommand;
let gpu = [];
const logEl = document.getElementById('log');
let targetOrder = [...graspObjects];
let slotOrderState = ['1', '2', '3', '4', '5', '6'];
let dragState = null;
const fixedBitongTargets = new Set(['bi']);
let placementPreviewTimer = null;
let placementPreviewSignature = '';

function buildChecks(containerId, names, checkedNames, prefix) {
  const checked = new Set(checkedNames || []);
  const root = document.getElementById(containerId);
  root.innerHTML = '';
  names.forEach((name) => {
    const label = document.createElement('label');
    label.className = 'check';
    label.dataset.name = name;
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.dataset.name = name;
    input.className = prefix;
    input.checked = checked.has(name);
    if (prefix === 'grasp-object') {
      input.onchange = () => {
        syncTargetOrderFromChecks();
        renderPlacementMapping();
      };
    }
    label.appendChild(input);
    label.appendChild(document.createTextNode(name));
    if (prefix === 'perception-object' || prefix === 'grasp-object') {
      const status = document.createElement('span');
      status.className = 'asset-status';
      label.appendChild(status);
    }
    root.appendChild(label);
  });
}

function selectedNames(cls) {
  return Array.from(document.querySelectorAll(`input.${cls}:checked`)).filter(x => !x.disabled).map(x => x.dataset.name);
}

function setChecks(cls, names) {
  const selected = new Set(names || []);
  document.querySelectorAll(`input.${cls}`).forEach((input) => { input.checked = selected.has(input.dataset.name) && !input.disabled; });
  if (cls === 'grasp-object') {
    syncTargetOrderFromChecks();
    renderPlacementMapping();
  }
}

function syncTargetOrderFromChecks() {
  const selected = new Set(selectedNames('grasp-object'));
  targetOrder = targetOrder.filter((name) => selected.has(name));
  graspObjects.forEach((name) => {
    if (selected.has(name) && !targetOrder.includes(name)) targetOrder.push(name);
  });
}

function selectedTargetsInOrder() {
  syncTargetOrderFromChecks();
  return [...targetOrder];
}

function targetSlotPairs() {
  const pairs = [];
  let slotIndex = 0;
  selectedTargetsInOrder().forEach((name) => {
    if (fixedBitongTargets.has(name)) {
      pairs.push({object: name, fixed: 'bitong'});
      return;
    }
    const slot = slotOrderState[slotIndex] || String(slotIndex + 1);
    slotIndex += 1;
    pairs.push({object: name, slot});
  });
  return pairs;
}

function sourceSlotMap() {
  return targetSlotPairs()
    .filter((item) => item.slot)
    .map((item) => `${item.object}:${item.slot}`);
}

function moveBefore(list, value, beforeValue) {
  const next = list.filter((item) => item !== value);
  const index = beforeValue ? next.indexOf(beforeValue) : -1;
  if (index < 0) next.push(value);
  else next.splice(index, 0, value);
  return next;
}

function makeDragCard(text, badge, listName, value, fixed=false) {
  const card = document.createElement('div');
  card.className = `drag-card${fixed ? ' fixed' : ''}`;
  card.dataset.list = listName;
  card.dataset.value = value;
  card.innerHTML = `<span>${text}</span><span class="badge">${badge || ''}</span>`;
  if (!fixed) {
    card.draggable = true;
    card.ondragstart = (ev) => {
      dragState = {list: listName, value};
      card.classList.add('dragging');
      ev.dataTransfer.effectAllowed = 'move';
      ev.dataTransfer.setData('text/plain', `${listName}:${value}`);
    };
    card.ondragend = () => {
      dragState = null;
      card.classList.remove('dragging');
    };
  }
  card.ondragover = (ev) => {
    if (dragState && dragState.list === listName && dragState.value !== value) ev.preventDefault();
  };
  card.ondrop = (ev) => {
    ev.preventDefault();
    if (!dragState || dragState.list !== listName || dragState.value === value) return;
    if (listName === 'targets') targetOrder = moveBefore(targetOrder, dragState.value, value);
    if (listName === 'slots') slotOrderState = moveBefore(slotOrderState, dragState.value, value);
    renderPlacementMapping();
  };
  return card;
}

function renderPlacementMapping() {
  const targetList = document.getElementById('targetCardList');
  const slotList = document.getElementById('slotCardList');
  const summary = document.getElementById('placementMappingText');
  if (!targetList || !slotList || !summary) return;
  const pairs = targetSlotPairs();
  targetList.innerHTML = '';
  pairs.forEach((item) => {
    const badge = item.fixed ? `→ ${item.fixed}` : `→ slot ${item.slot}`;
    targetList.appendChild(makeDragCard(item.object, badge, 'targets', item.object));
  });
  if (!pairs.length) {
    const empty = document.createElement('div');
    empty.className = 'note';
    empty.textContent = '未选择抓取目标';
    targetList.appendChild(empty);
  }
  slotList.innerHTML = '';
  const usedSlots = new Set();
  pairs.forEach((item) => {
    if (item.fixed) {
      slotList.appendChild(makeDragCard('bitong', 'bi 固定', 'fixed', `fixed-${item.object}`, true));
      return;
    }
    usedSlots.add(item.slot);
    slotList.appendChild(makeDragCard(`slot ${item.slot}`, item.object, 'slots', item.slot));
  });
  slotOrderState.filter((slot) => !usedSlots.has(slot)).forEach((slot) => {
    slotList.appendChild(makeDragCard(`slot ${slot}`, '未用', 'slots', slot));
  });
  const mapped = sourceSlotMap();
  const fixed = pairs.filter((item) => item.fixed).map((item) => `${item.object}→${item.fixed}`);
  summary.textContent = `桌面映射：${mapped.length ? mapped.join('，') : '无'}${fixed.length ? `；固定：${fixed.join('，')}` : ''}`;
  renderSequencePreview(buildLocalMappingPreview());
  schedulePlacementPreview();
}

function taskConfig() {
  const executionMode = document.getElementById('executionMode').value;
  return {
    objects: selectedTargetsInOrder(),
    tracked_objects: trackedObjects,
    slot_order: slotOrderState,
    source_slot_map: sourceSlotMap(),
    random_targets: document.getElementById('randomTargets').checked,
    execute_real: executionMode === 'real',
    execution_mode: executionMode,
    real_control_hz: Number(document.getElementById('realHz').value || 30),
    real_max_delta_per_step: Number(document.getElementById('realDelta').value || 0.1),
    confirm_segmentation: false,
    reuse_latest_perception: true,
    render_mode: executionMode === 'dry' ? document.getElementById('renderMode').value : document.getElementById('renderMode').value,
  };
}

function perceptionConfig() {
  return {
    object_names: selectedNames('perception-object'),
    confirm_segmentation: true,
  };
}

function initControls() {
  buildChecks('graspObjectChecks', graspObjects, graspObjects, 'grasp-object');
  buildChecks('perceptionObjectChecks', allSceneObjects, allSceneObjects, 'perception-object');
  syncTargetOrderFromChecks();
  renderPlacementMapping();
}

function appendLog(line, cls='') {
  const time = new Date().toLocaleTimeString();
  logEl.textContent += `[${time}] ${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function stateText(item) {
  if (!item) return '<span class="warn">未知</span>';
  if (item.phase === 'error') {
    const err = item.last_error ? String(item.last_error).slice(0, 160) : '未知错误';
    return `<span class="bad">失败</span><small>${err}</small>`;
  }
  if (item.name === '分割定位' && item.phase === 'stopped' && item.returncode === 0) {
    return `<span class="ok">定位完成</span><small>${perceptionDetail(item)}</small>`;
  }
  if (item.ready) {
    return `<span class="ok">热启动成功 pid=${item.pid || '-'}</span><small>${hotDetail(item)}</small>`;
  }
  if (item.phase === 'capturing') return `<span class="warn">拍照中</span>`;
  if (item.phase === 'segmenting') return `<span class="warn">SAM3 分割中</span>`;
  if (item.phase === 'posing') return `<span class="warn">SAM6D 定位中</span>`;
  if (item.phase === 'warming') return `<span class="warn">热启动中 pid=${item.pid || '-'}</span>`;
  if (item.phase === 'starting' || item.phase === 'started') return `<span class="warn">常驻已启动 pid=${item.pid || '-'}</span><small>等待热启动完成</small>`;
  if (item.running) return `<span class="warn">运行中 pid=${item.pid}</span>`;
  if (item.returncode === null || item.returncode === undefined) return '<span class="warn">未启动</span>';
  return `<span class="bad">已退出 code=${item.returncode}</span>`;
}

function fmtMs(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '?';
  return n >= 1000 ? `${(n / 1000).toFixed(2)}s` : `${n.toFixed(0)}ms`;
}

function hotDetail(item) {
  const r = item.last_result || {};
  const parts = [];
  if (r.elapsed_ms !== undefined) parts.push(`加载 ${fmtMs(r.elapsed_ms)}`);
  if (r.image_warmup_ms !== undefined) parts.push(`图像 ${fmtMs(r.image_warmup_ms)}`);
  if (r.model_warmup_ms !== undefined) parts.push(`模型 ${fmtMs(r.model_warmup_ms)}`);
  if (r.template_features) {
    parts.push(`模板 ${r.template_features.ok_count || 0}/${r.template_features.object_count || 0} ${fmtMs(r.template_features.elapsed_ms)}`);
  }
  return parts.join(' · ') || '完成';
}

function perceptionDetail(item) {
  const r = item.last_result || {};
  const parts = [];
  if (r.ok_count !== undefined && r.object_count !== undefined) parts.push(`ok ${r.ok_count}/${r.object_count}`);
  const found = item.mask_found_objects || r.mask_found_objects || [];
  const missing = item.mask_missing_objects || r.mask_missing_objects || [];
  if (found.length || missing.length) parts.push(`mask ${found.length}/${found.length + missing.length}`);
  if (r.elapsed_ms !== undefined) parts.push(`PEM ${fmtMs(r.elapsed_ms)}`);
  if (r.result_path) parts.push(String(r.result_path).split('/').slice(-2).join('/'));
  return parts.join(' · ') || '完成';
}

function updatePerceptionAssetMarks(data) {
  const perception = data.perception || {};
  const latest = data.latest_perception_result || {};
  const result = perception.last_result || {};
  const found = new Set(perception.mask_found_objects || result.mask_found_objects || latest.mask_found_objects || []);
  const missing = new Set(perception.mask_missing_objects || result.mask_missing_objects || latest.mask_missing_objects || []);
  const poseFound = new Set(perception.pose_found_objects || result.pose_found_objects || latest.pose_found_objects || []);
  const poseMissing = new Set(perception.pose_missing_objects || result.pose_missing_objects || latest.pose_missing_objects || []);
  const hasPerceptionMarks = found.size > 0 || missing.size > 0 || poseFound.size > 0 || poseMissing.size > 0;
  document.querySelectorAll('input.perception-object').forEach((input) => {
    const label = input.closest('label.check');
    if (!label) return;
    const status = label.querySelector('.asset-status');
    label.classList.remove('asset-found', 'asset-missing');
    const name = input.dataset.name;
    if (found.has(name)) {
      label.classList.add('asset-found');
      if (status) status.textContent = '找到';
    } else if (missing.has(name)) {
      label.classList.add('asset-missing');
      if (status) status.textContent = '缺失';
    } else if (status) {
      status.textContent = '';
    }
  });
  let changed = false;
  document.querySelectorAll('input.grasp-object').forEach((input) => {
    const label = input.closest('label.check');
    if (!label) return;
    const status = label.querySelector('.asset-status');
    const name = input.dataset.name;
    label.classList.remove('asset-found', 'asset-missing');
    input.dataset.perceptionDisabled = '0';
    input.disabled = false;
    if (poseFound.has(name) || (!poseFound.size && found.has(name))) {
      label.classList.add('asset-found');
      if (status) status.textContent = '可抓';
    } else if (missing.has(name)) {
      label.classList.add('asset-missing');
      if (status) status.textContent = '未分割';
      if (input.checked) changed = true;
      input.checked = false;
      input.dataset.perceptionDisabled = '1';
      input.disabled = true;
    } else if (poseMissing.has(name) || hasPerceptionMarks) {
      label.classList.add('asset-missing');
      if (status) status.textContent = '未定位';
      if (input.checked) changed = true;
      input.checked = false;
      input.dataset.perceptionDisabled = '1';
      input.disabled = true;
    } else if (status) {
      status.textContent = '';
    }
  });
  if (changed) {
    syncTargetOrderFromChecks();
    renderPlacementMapping();
  }
}

function levelLabel(level) {
  if (level === 'bad') return '阻断';
  if (level === 'warn') return '注意';
  return '通过';
}

function renderPreflight(report) {
  const list = document.getElementById('preflightList');
  const metric = document.getElementById('preflightMetric');
  if (!list || !report) return;
  list.innerHTML = '';
  (report.checks || []).forEach((item) => {
    const div = document.createElement('div');
    div.className = `preflight-item ${item.level || 'warn'}`;
    div.innerHTML = `<b>${levelLabel(item.level)} · ${item.title || ''}</b><br><small>${item.detail || ''}</small>`;
    list.appendChild(div);
  });
  if (!(report.checks || []).length) {
    list.innerHTML = '<div class="preflight-item warn"><b>未检查</b><br><small>等待运行前检查</small></div>';
  }
  if (metric) {
    metric.textContent = report.ok ? '通过' : '未通过';
    metric.style.color = report.ok ? 'var(--green)' : 'var(--red)';
  }
  renderSequencePreview(report.mapping || buildLocalMappingPreview());
}

function buildLocalMappingPreview() {
  return targetSlotPairs().map((item, index) => ({
    index: index + 1,
    object: item.object,
    destination: item.fixed || `slot_${item.slot}`,
    fixed: Boolean(item.fixed),
  }));
}

function renderSequencePreview(mapping) {
  const root = document.getElementById('sequencePreview');
  if (!root) return;
  root.innerHTML = '';
  (mapping || []).forEach((item) => {
    const div = document.createElement('div');
    div.className = 'sequence-item';
    div.innerHTML = `<b>${item.index}. ${item.object}</b><small>${item.destination || '未分配'}</small>`;
    root.appendChild(div);
  });
  if (!(mapping || []).length) {
    root.innerHTML = '<div class="sequence-item"><b>暂无目标</b><small>选择抓取目标后显示</small></div>';
  }
}

async function runPreflight(showLog=true) {
  const data = await postJSON('/api/preflight', {config: taskConfig()});
  renderPreflight(data.report);
  if (showLog) appendLog(data.report.ok ? '运行前检查通过' : '运行前检查未通过');
  return data.report;
}

function placementPreviewKey() {
  const latest = (window.lastStatusData || {}).latest_perception_result || {};
  if (!latest.result_path) return '';
  return JSON.stringify({
    result_path: latest.result_path,
    objects: selectedTargetsInOrder(),
    source_slot_map: sourceSlotMap(),
  });
}

function schedulePlacementPreview() {
  const key = placementPreviewKey();
  if (!key || key === placementPreviewSignature) return;
  if (placementPreviewTimer) clearTimeout(placementPreviewTimer);
  placementPreviewTimer = setTimeout(() => {
    runPlacementPreview(false).catch(() => {});
  }, 650);
}

async function runPlacementPreview(showLog=true) {
  const key = placementPreviewKey();
  if (!key) {
    if (showLog) appendLog('还没有最近定位结果，无法生成放置预览');
    return null;
  }
  const data = await postJSON('/api/placement-preview', {config: taskConfig()});
  placementPreviewSignature = key;
  if (showLog) appendLog(`放置预览已更新：${data.rel || data.path}`);
  await refreshImages();
  return data;
}

function renderTargetStatus(data) {
  const root = document.getElementById('targetStatusGrid');
  if (!root) return;
  const perception = data.perception || {};
  const latest = data.latest_perception_result || {};
  const result = perception.last_result || {};
  const maskFound = new Set(perception.mask_found_objects || result.mask_found_objects || latest.mask_found_objects || []);
  const maskMissing = new Set(perception.mask_missing_objects || result.mask_missing_objects || latest.mask_missing_objects || []);
  const poseFound = new Set(perception.pose_found_objects || result.pose_found_objects || latest.pose_found_objects || []);
  const poseMissing = new Set(perception.pose_missing_objects || result.pose_missing_objects || latest.pose_missing_objects || []);
  const details = latest.pose_details || result.pose_details || {};
  root.innerHTML = '';
  allSceneObjects.forEach((name) => {
    const d = details[name] || {};
    let cls = 'warn';
    let state = '未检测';
    if (poseFound.has(name)) { cls = 'ok'; state = '定位成功'; }
    else if (poseMissing.has(name)) { cls = 'bad'; state = '定位失败'; }
    else if (maskFound.has(name)) { cls = 'warn'; state = '已分割'; }
    else if (maskMissing.has(name)) { cls = 'bad'; state = '未分割'; }
    const score = d.score !== undefined && d.score !== null ? `score ${Number(d.score).toFixed(3)}` : '';
    const refine = d.refine_applied === true ? 'refine 已用' : d.refine_applied === false ? 'refine 未用' : '';
    const div = document.createElement('div');
    div.className = `target-card ${cls}`;
    div.innerHTML = `<b>${name}</b><small>${state}${score ? ' · ' + score : ''}${refine ? ' · ' + refine : ''}</small>`;
    root.appendChild(div);
  });
}

function renderTimeline(data) {
  const root = document.getElementById('stageTimeline');
  if (!root) return;
  const phases = [
    ['hot', '热启动'],
    ['capture', '拍照'],
    ['seg', 'SAM3'],
    ['pose', 'SAM6D'],
    ['plan', '规划'],
    ['motion', '执行'],
    ['done', '完成'],
  ];
  const perception = data.perception || {};
  const grasp = data.grasp || {};
  const samReady = data.sam3?.ready && data.sam6d?.ready;
  root.innerHTML = '';
  phases.forEach(([key, label]) => {
    let cls = '';
    if (key === 'hot' && samReady) cls = 'done';
    if (key === 'capture' && ['capturing','segmenting','posing','stopped'].includes(perception.phase)) cls = perception.phase === 'capturing' ? 'active' : 'done';
    if (key === 'seg' && ['segmenting','posing','stopped'].includes(perception.phase)) cls = perception.phase === 'segmenting' ? 'active' : 'done';
    if (key === 'pose' && ['posing','stopped'].includes(perception.phase)) cls = perception.phase === 'posing' ? 'active' : 'done';
    if (key === 'plan' && grasp.running) cls = 'active';
    if (key === 'motion' && grasp.running) cls = 'active';
    if (perception.phase === 'error' && ['capture','seg','pose'].includes(key)) cls = 'bad';
    const div = document.createElement('div');
    div.className = `timeline-step ${cls}`;
    div.textContent = label;
    root.appendChild(div);
  });
}

function renderSafety(data) {
  const cfg = taskConfig();
  const modeText = cfg.execute_real ? '真机执行' : (cfg.execution_mode === 'sim' ? '仿真窗口' : '只规划');
  document.getElementById('safetyMode').textContent = modeText;
  document.getElementById('realParamText').textContent = `${cfg.real_control_hz}Hz / ${cfg.real_max_delta_per_step}`;
  const latest = data.latest_perception_result || {};
  document.getElementById('perceptionMetric').textContent = latest.result_path ? `${latest.ok_count || 0}/${latest.object_count || 0}` : '暂无';
  const pill = document.getElementById('modePill');
  pill.textContent = data.grasp?.running ? '抓取运行中' : modeText;
  pill.className = `pill ${cfg.execute_real ? 'warn' : 'ok'}`;
}

function setUiLocked(locked) {
  const ids = ['hotAllBtn','hotSam3Btn','hotSam6dBtn','selectAllGraspBtn','clearGraspBtn','randomTargets','executionMode','realHz','realDelta','renderMode','runPerceptionBtn','syncPerceptionBtn','buildCmdBtn','startConfiguredBtn','startBtn'];
  ids.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = Boolean(locked);
  });
  document.querySelectorAll('input.grasp-object, input.perception-object').forEach((el) => {
    const perceptionDisabled = el.dataset.perceptionDisabled === '1';
    el.disabled = Boolean(locked) || perceptionDisabled;
  });
}

function renderFailure(failure) {
  const root = document.getElementById('failureCard');
  if (!root) return;
  if (!failure || !failure.has_failure) {
    root.className = 'failure-card';
    root.textContent = '暂无失败记录';
    return;
  }
  root.className = 'failure-card bad';
  const msg = (failure.messages || [])[0];
  const img = failure.latest_image;
  root.innerHTML = `<b>${msg ? '最近失败' : '最近失败截图'}</b><br><small>${msg ? msg.message : ''}</small>${img ? `<br><small>${img.rel}</small>` : ''}`;
}

function drawWaterfall(profile) {
  const canvas = document.getElementById('waterfallChart');
  const text = document.getElementById('profileText');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#fcfcfd';
  ctx.fillRect(0, 0, w, h);
  const stages = (profile?.stages || []).slice(0, 10);
  if (!stages.length) {
    ctx.fillStyle = '#66717f';
    ctx.fillText('暂无 profile', 20, 30);
    if (text) text.textContent = '等待 planning_profile...';
    return;
  }
  const left = 170, top = 18, rowH = 22, barW = w - left - 32;
  const maxMs = Math.max(...stages.map(x => Number(x.total_ms || 0)), 1);
  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  stages.forEach((item, idx) => {
    const y = top + idx * rowH;
    const value = Number(item.total_ms || 0);
    const width = Math.max(2, barW * value / maxMs);
    ctx.fillStyle = '#475569';
    ctx.textAlign = 'right';
    ctx.fillText(String(item.stage || '').slice(0, 24), left - 8, y + 13);
    ctx.fillStyle = idx === 0 ? '#2563eb' : idx < 3 ? '#0891b2' : '#94a3b8';
    ctx.fillRect(left, y + 4, width, 14);
    ctx.fillStyle = '#0f172a';
    ctx.textAlign = 'left';
    ctx.fillText(fmtMs(value), left + width + 6, y + 13);
  });
  if (text) {
    const path = profile.path ? String(profile.path).split('/').slice(-2).join('/') : '';
    text.textContent = `profile ${fmtMs(profile.total_ms || 0)} · ${path}`;
  }
}

async function postJSON(url, body={}) {
  const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || JSON.stringify(data));
  return data;
}

async function refreshStatus() {
  const data = await (await fetch('/api/status')).json();
  window.lastStatusData = data;
  document.getElementById('serverStatus').textContent = `server pid=${data.server_pid} rss=${data.rss_mb}MB`;
  document.getElementById('runMeta').textContent = `run ${data.run_id || '-'} · started ${new Date((data.server_started_at || 0) * 1000).toLocaleString()}`;
  document.getElementById('sam3State').innerHTML = stateText(data.sam3);
  document.getElementById('sam6dState').innerHTML = stateText(data.sam6d);
  document.getElementById('perceptionState').innerHTML = stateText(data.perception);
  document.getElementById('graspState').innerHTML = stateText(data.grasp);
  updatePerceptionAssetMarks(data);
  renderTargetStatus(data);
  renderTimeline(data);
  renderSafety(data);
  renderFailure(data.failure);
  drawWaterfall(data.profile);
  setUiLocked(Boolean(data.grasp && data.grasp.running));
  const latest = data.latest_perception_result || {};
  const latestText = latest.result_path
    ? `最近定位：ok ${latest.ok_count || 0}/${latest.object_count || 0} · ${String(latest.result_path).split('/').slice(-2).join('/')}`
    : '最近定位：暂无。正式抓取默认复用这里的结果，不会重新分割定位。';
  document.getElementById('latestPerceptionText').textContent = latestText;
  schedulePlacementPreview();
  gpu = data.gpu || gpu;
  drawGpu();
}

async function refreshImages() {
  const data = await (await fetch('/api/latest-images')).json();
  setImg('sam3', data.sam3);
  setImg('pem', data.pem);
  setImg('placement', data.placement);
  setImg('rgb', data.rgb);
  setImg('sapien', data.sapien);
}

function setImg(id, list) {
  const img = document.getElementById(id + 'Img');
  const cap = document.getElementById(id + 'Caption');
  const empty = document.getElementById(id + 'Empty');
  if (!list || !list.length) {
    cap.textContent = cap.textContent.split(' | ')[0] + ' | 暂无';
    img.removeAttribute('src');
    img.style.display = 'none';
    if (empty) empty.style.display = 'flex';
    return;
  }
  const item = list[0];
  cap.textContent = `${cap.textContent.split(' | ')[0]} | ${item.rel}`;
  img.src = `/image?path=${encodeURIComponent(item.path)}&t=${Date.now()}`;
  img.style.display = 'block';
  if (empty) empty.style.display = 'none';
}

function drawGpu() {
  const canvas = document.getElementById('gpuChart');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const left = 44, right = 54, top = 18, bottom = 30;
  const plotW = w - left - right;
  const plotH = h - top - bottom;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#fcfcfd';
  ctx.fillRect(0,0,w,h);

  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textBaseline = 'middle';
  ctx.strokeStyle = '#d8dde3';
  ctx.fillStyle = '#6b7280';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const ratio = i / 5;
    const y = top + plotH * ratio;
    const utilTick = Math.round(100 * (1 - ratio));
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + plotW, y);
    ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(`${utilTick}%`, left - 6, y);
  }
  ctx.strokeStyle = '#9ca3af';
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, top + plotH);
  ctx.lineTo(left + plotW, top + plotH);
  ctx.lineTo(left + plotW, top);
  ctx.stroke();

  const samples = (gpu || []).filter(x => x.ok).slice(-180);
  const maxMem = samples.length ? Math.max(...samples.map(x => x.mem_total || 1)) : 8192;
  for (let i = 0; i <= 4; i++) {
    const ratio = i / 4;
    const y = top + plotH * (1 - ratio);
    const memTick = maxMem * ratio;
    ctx.textAlign = 'left';
    ctx.fillStyle = '#6b7280';
    ctx.fillText(memTick >= 1024 ? `${(memTick / 1024).toFixed(1)}G` : `${memTick.toFixed(0)}M`, left + plotW + 6, y);
  }
  ctx.textAlign = 'left';
  ctx.fillStyle = '#6b7280';
  ctx.fillText('最近 180s', left, h - 10);
  ctx.textAlign = 'right';
  ctx.fillText('现在', left + plotW, h - 10);
  ctx.fillStyle = '#2563eb';
  ctx.fillText('GPU利用率(左轴)', left + 104, 10);
  ctx.fillStyle = '#15803d';
  ctx.fillText('显存(右轴)', left + 190, 10);
  if (!samples.length) return;

  function plot(color, values, maxVal) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = left + plotW * i / Math.max(values.length - 1, 1);
      const y = top + plotH * (1 - Math.max(0, Math.min(v / maxVal, 1)));
      if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
  }
  plot('#2563eb', samples.map(x => x.util || 0), 100);
  plot('#15803d', samples.map(x => x.mem_used || 0), maxMem);
  const last = samples[samples.length-1];
  document.getElementById('gpuText').textContent =
    `GPU ${last.util.toFixed(0)}% | 显存 ${last.mem_used.toFixed(0)}/${last.mem_total.toFixed(0)} MB | 温度 ${last.temp.toFixed(0)}°C | 功耗 ${last.power.toFixed(1)}W`;
}

document.getElementById('hotAllBtn').onclick = async () => { appendLog('请求热启动 SAM3 + SAM6D'); await postJSON('/api/hotstart/all'); };
document.getElementById('hotSam3Btn').onclick = async () => { appendLog('请求热启动 SAM3'); await postJSON('/api/hotstart/sam3'); };
document.getElementById('hotSam6dBtn').onclick = async () => { appendLog('请求热启动 SAM6D'); await postJSON('/api/hotstart/sam6d'); };
document.getElementById('stopHotBtn').onclick = async () => { await postJSON('/api/hotstart/stop'); };
document.getElementById('selectAllGraspBtn').onclick = () => setChecks('grasp-object', graspObjects);
document.getElementById('clearGraspBtn').onclick = () => setChecks('grasp-object', []);
document.getElementById('syncPerceptionBtn').onclick = () => setChecks('perception-object', [...selectedTargetsInOrder(), ...trackedObjects]);
document.getElementById('preflightBtn').onclick = async () => { await runPreflight(true); };
document.getElementById('placementPreviewBtn').onclick = async () => { await runPlacementPreview(true); };
document.getElementById('debugPackBtn').onclick = async () => {
  const data = await postJSON('/api/debug-pack', {});
  document.getElementById('debugPackText').textContent = `调试包：${data.path} (${data.file_count} files)`;
  appendLog(`调试包已生成：${data.path}`);
};
['executionMode','realHz','realDelta','randomTargets','renderMode'].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.onchange = () => { renderSafety(window.lastStatusData || {}); renderSequencePreview(buildLocalMappingPreview()); };
});
document.getElementById('runPerceptionBtn').onclick = async () => {
  appendLog('启动一次分割定位，不执行抓取；复用常驻 SAM3/SAM6D');
  await postJSON('/api/perception/run', {config: perceptionConfig()});
};
document.getElementById('stopPerceptionBtn').onclick = async () => { await postJSON('/api/perception/stop'); };
document.getElementById('buildCmdBtn').onclick = async () => {
  const data = await postJSON('/api/grasp/command', {config: taskConfig()});
  document.getElementById('command').value = data.command;
  appendLog('已生成当前配置的命令预览；开始抓取不依赖这一步');
};
document.getElementById('startConfiguredBtn').onclick = async () => {
  const config = taskConfig();
  const report = await runPreflight(true);
  if (!report.ok) return;
  if (config.execute_real && !window.confirm('确认开始真机执行？')) return;
  const data = await postJSON('/api/grasp/start', {config});
  document.getElementById('command').value = data.command || document.getElementById('command').value;
};
document.getElementById('startBtn').onclick = async () => {
  if (!window.confirm('命令框是高级调试入口。正常使用请点“按配置开始抓取”。确认运行命令框内容？')) return;
  const data = await postJSON('/api/grasp/start', {command: document.getElementById('command').value});
  if (data.command) document.getElementById('command').value = data.command;
};
document.getElementById('stopBtn').onclick = async () => { await postJSON('/api/grasp/stop'); };
document.getElementById('enterBtn').onclick = async () => { await postJSON('/api/grasp/stdin', {text: '\n'}); };
document.getElementById('retryBtn').onclick = async () => { await postJSON('/api/grasp/stdin', {text: 'r\n'}); };
document.getElementById('quitBtn').onclick = async () => { await postJSON('/api/grasp/stdin', {text: 'q\n'}); };
document.getElementById('refreshBtn').onclick = () => { refreshStatus(); refreshImages(); };

const es = new EventSource('/events');
es.onmessage = (ev) => {
  const item = JSON.parse(ev.data);
  if (item.kind === 'gpu') {
    gpu.push(item); if (gpu.length > 240) gpu.shift(); drawGpu(); return;
  }
  if (item.kind === 'summary' || item.kind === 'status' || item.kind === 'hotstart') {
    appendLog(item.message);
    if (item.kind === 'hotstart') refreshStatus();
  }
  else if (item.kind === 'process') appendLog(`${item.process}/${item.stream}: ${item.message}`);
};
setInterval(refreshStatus, 2000);
setInterval(refreshImages, 2500);
initControls();
renderPreflight({ok:false, checks:[], mapping:buildLocalMappingPreview()});
refreshStatus();
refreshImages();
</script>
</body>
</html>""".replace("__DEFAULT_COMMAND_JSON__", json.dumps(DEFAULT_GRASP_COMMAND, ensure_ascii=False)).replace(
    "__GRASP_OBJECTS_JSON__", json.dumps(DEFAULT_GRASP_OBJECTS, ensure_ascii=False)
).replace(
    "__TRACKED_OBJECTS_JSON__", json.dumps(DEFAULT_TRACKED_OBJECTS, ensure_ascii=False)
).replace(
    "__ALL_OBJECTS_JSON__", json.dumps(DEFAULT_OBJECTS, ensure_ascii=False)
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Web control panel for RM75 SAM6D grasp pipeline.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    emit("status", f"Web 控制台启动 http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
