import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
import tkinter as tk
from tkinter import ttk

from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils import common
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from lerobot_sim2real.config.real_robot import create_real_robot
from lerobot_sim2real.utils.safety import setup_safe_exit


class ClippedLeRobotRealAgent(LeRobotRealAgent):
    """LeRobot agent that trims extra simulated joints before commanding the real robot."""

    def __init__(self, robot, use_cached_qpos=True, max_retries=3, retry_delay=0.1, **kwargs):
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._gripper_limits_rad = None
        super().__init__(robot, use_cached_qpos=use_cached_qpos, **kwargs)
        self._uses_le_robot_bus = hasattr(self.real_robot, "bus")

    def set_gripper_limits(self, limits_rad):
        self._gripper_limits_rad = tuple(float(x) for x in limits_rad)

    def _ensure_realman_motor_keys(self):
        if self._motor_keys is not None:
            return
        try:
            obs = self.real_robot.get_observation()
        except Exception:
            return
        ordered = [k for k in obs.keys() if k.endswith(".pos")]
        if ordered:
            self._motor_keys = ordered

    def _trim_qpos(self, qpos):
        tensor = torch.as_tensor(qpos, dtype=torch.float32)
        if tensor.ndim > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        try:
            if self._uses_le_robot_bus:
                self._ensure_motor_keys()
            else:
                self._ensure_realman_motor_keys()
        except Exception:
            # If keys aren't ready yet, just keep the tensor as-is; parent will raise if needed.
            return tensor.squeeze(0)
        max_dof = len(self._motor_keys or [])
        if max_dof > 0 and tensor.shape[-1] > max_dof:
            tensor = tensor[..., :max_dof]
        return tensor.squeeze(0)

    def _call_with_retry(self, fn, *args, **kwargs):
        last_exc = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    break
                print(f"[RealRobot Retry] attempt {attempt}/{self._max_retries} failed: {exc}. Retrying...")
                time.sleep(self._retry_delay)
        raise last_exc

    def set_target_qpos(self, qpos):
        qpos_tensor = common.to_cpu_tensor(qpos).flatten()
        if self._uses_le_robot_bus:
            gripper_value = None
            if self._gripper_limits_rad is not None:
                self._ensure_motor_keys()
                arm_dof = len(self._motor_keys or [])
                if qpos_tensor.shape[0] > arm_dof:
                    gripper_value = float(qpos_tensor[-1])
                    qpos_tensor = qpos_tensor[:arm_dof]

            def _attempt():
                return super(ClippedLeRobotRealAgent, self).set_target_qpos(qpos_tensor)

            result = self._call_with_retry(_attempt)

            if gripper_value is not None:
                lo, hi = self._gripper_limits_rad
                span = max(hi - lo, 1e-6)
                norm = (gripper_value - lo) / span
                norm = float(np.clip(norm, 0.0, 1.0))
                try:
                    self.real_robot.send_action({"gripper.pos": norm})
                except Exception as exc:
                    print(f"[WARN] failed to send gripper command: {exc}")
            return result

        self._ensure_realman_motor_keys()
        if not self._motor_keys:
            raise RuntimeError("Unable to infer real robot joint names.")

        action = {}
        for i, key in enumerate(self._motor_keys):
            if i >= qpos_tensor.shape[0]:
                break
            value = float(qpos_tensor[i])
            if key.startswith("gripper") and self._gripper_limits_rad is not None:
                lo, hi = self._gripper_limits_rad
                span = max(hi - lo, 1e-6)
                value = (value - lo) / span
                value = float(np.clip(value, 0.0, 1.0))
            else:
                if getattr(self.real_robot.config, "use_degrees", False):
                    value = float(np.rad2deg(value))
            action[key] = value

        def _attempt_realman():
            return self.real_robot.send_action(action)

        return self._call_with_retry(_attempt_realman)

    def reset(self, qpos):
        trimmed = self._trim_qpos(qpos)
        if self._uses_le_robot_bus:
            def _attempt():
                return super(ClippedLeRobotRealAgent, self).reset(trimmed)

            return self._call_with_retry(_attempt)

        target = self.get_qpos().detach().cpu().flatten()
        desired = common.to_cpu_tensor(trimmed).flatten()
        freq = 30
        max_rad_per_step = 0.025
        for _ in range(int(20 * freq)):
            start_t = time.perf_counter()
            delta = (desired - target).clamp(
                min=-max_rad_per_step, max=max_rad_per_step
            )
            if torch.norm(delta).item() <= 1e-4:
                break
            target += delta
            self.set_target_qpos(target)
            dt = time.perf_counter() - start_t
            remaining = max(0.0, 1 / freq - dt)
            time.sleep(remaining)

        return target

    def get_qpos(self):
        if self._uses_le_robot_bus:
            return super().get_qpos()
        obs = self.real_robot.get_observation()
        ordered = [k for k in obs.keys() if k.endswith(".pos")]
        if not ordered:
            raise RuntimeError("Real robot observation missing joint positions.")
        self._motor_keys = ordered

        values = []
        for key in ordered:
            val = float(obs[key])
            if key.startswith("gripper") and self._gripper_limits_rad is not None:
                lo, hi = self._gripper_limits_rad
                span = max(hi - lo, 1e-6)
                val = lo + val * span
            else:
                if getattr(self.real_robot.config, "use_degrees", False):
                    val = float(np.deg2rad(val))
            values.append(val)
        qpos = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        self._cached_qpos = qpos
        return qpos

    def apply_sim_qpos(self, sim_qpos, override_gripper_norm=None):
        sim_qpos = np.asarray(sim_qpos, dtype=np.float32).reshape(-1)
        drive_limit = getattr(self, "_gripper_limits_rad", None)
        if drive_limit is not None:
            lo, hi = drive_limit
        else:
            lo, hi = 0.0, 0.91
        span = max(hi - lo, 1e-6)
        drive_idx = getattr(self, "_gripper_drive_index", None)
        if drive_idx is None:
            drive_idx = getattr(self.real_robot, "_gripper_drive_index", None)
        if drive_idx is None or drive_idx >= sim_qpos.shape[0]:
            drive_idx = sim_qpos.shape[0] - 1
        if override_gripper_norm is not None:
            gripper_val = float(np.clip(override_gripper_norm, 0.0, 1.0))
        else:
            gripper_val = float(np.clip((sim_qpos[drive_idx] - lo) / span, 0.0, 1.0))
        if self._motor_keys is None:
            self._ensure_realman_motor_keys()
        arm_len = max(0, (len(self._motor_keys or []) - 1))
        arm_vals = sim_qpos[:arm_len]
        if getattr(self.real_robot.config, "use_degrees", False):
            arm_vals = np.rad2deg(arm_vals)
        action = {}
        for i in range(min(len(arm_vals), len(self._motor_keys or []))):
            action[self._motor_keys[i]] = float(arm_vals[i])  # ✅ key 已经自带 .pos
        action["gripper.pos"] = gripper_val
        self.real_robot.send_action(action)
        print("send_action keys:", len(action), list(action.keys())[:5], "...")


def to_numpy_image(image):
    if torch.is_tensor(image):
        return image.detach().cpu().numpy().astype(np.float32)
    return np.asarray(image, dtype=np.float32)


@dataclass
class Args:
    env_id: str = f"RM75GraspCube_two_cameras-v1"
    """The environment id to train on"""
    env_kwargs_json_path: Optional[str] = "../../so101_env_config.json"
    """Path to a json file containing additional environment kwargs to use."""
    auto_sync_on_start: bool = True
    """Whether to automatically sync real qpos into sim once at startup."""
    max_step_deg: float = 30.0
    """Max joint delta (degrees) applied per control tick when chasing slider targets."""
    sync_tol_deg: float = 0.25
    """Stop commanding when real-vs-target error falls below this tolerance."""
    enable_gripper_control: bool = True
    """Expose gripper slider/action if the environment/controller supports it."""

def overlay_envs(sim_env, real_env):
    """
    Overlays sim_env observtions onto real_env observations
    Requires matching ids between the two environments' sensors
    e.g. id=phone_camera sensor in real_env / real_robot config, must have identical id in sim_env
    """
    real_obs = real_env.get_obs()["sensor_data"]
    sim_obs = sim_env.get_obs()["sensor_data"]
    assert sorted(real_obs.keys()) == sorted(
        sim_obs.keys()
    ), f"real camera names {real_obs.keys()} and sim camera names {sim_obs.keys()} differ"

    # overlaid_dict = sim_env.get_obs()["sensor_data"]
    overlaid_imgs = []
    sim_imgs_list = []
    real_imgs_list = []
    for name in sim_obs:  # ✅ 直接用 sim_obs 做 overlay
        real_imgs = to_numpy_image(real_obs[name]["rgb"][0]) / 255.0
        sim_imgs = to_numpy_image(sim_obs[name]["rgb"][0]) / 255.0
        overlaid_imgs.append(0.5 * real_imgs + 0.5 * sim_imgs)
        sim_imgs_list.append(sim_imgs)
        real_imgs_list.append(real_imgs)
        loss = np.mean((real_imgs - sim_imgs) ** 2)
        # print("current loss", loss)

    return tile_images(overlaid_imgs), tile_images(sim_imgs_list), tile_images(real_imgs_list)
def pad_qpos(values, dim):
    """Pad/trim a 1D arraylike to the desired dim."""
    arr = np.zeros(dim, dtype=np.float32)
    if values is None:
        return arr
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    length = min(dim, flat.shape[0])
    arr[:length] = flat[:length]
    return arr



def _is_normalized_action_space(action_space, dof: int) -> bool:
    """Heuristic: many ManiSkill joint controllers use normalized [-1,1] action space."""
    try:
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)[:dof]
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)[:dof]
        return np.allclose(low, -1.0, atol=1e-3) and np.allclose(high, 1.0, atol=1e-3)
    except Exception:
        return False


def _sanitize_joint_limits_rad(joint_limits_rad: np.ndarray, dof: int) -> np.ndarray:
    """Ensure finite, non-degenerate limits for mapping/clamping."""
    jl = np.asarray(joint_limits_rad, dtype=np.float32).reshape(-1, 2)[:dof].copy()
    for i in range(jl.shape[0]):
        lo, hi = float(jl[i, 0]), float(jl[i, 1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
            # fallback to [-pi, pi]
            jl[i, 0] = -np.pi
            jl[i, 1] = np.pi
    return jl


def infer_action_transform(real_env, action_space, action_dim: int, dof: int):
    """
    Try to infer how env.action maps to joint motion by inspecting ManiSkill controller attributes.
    Returns (mode, scale, offset, controller_obj):
      - mode: 'delta' or 'absolute'
      - scale/offset: np.ndarray length >= dof (float32), best-effort
    """
    ctrl = None
    try:
        ctrl = real_env.base_sim_env.agent.controller
    except Exception:
        try:
            ctrl = real_env.agent.controller
        except Exception:
            ctrl = None

    scale = None
    offset = None
    if ctrl is not None:
        for attr in ("action_scale", "act_scale", "scale"):
            if hasattr(ctrl, attr):
                scale = getattr(ctrl, attr)
                break
        for attr in ("action_offset", "act_offset", "offset"):
            if hasattr(ctrl, attr):
                offset = getattr(ctrl, attr)
                break

    def _to_np(x, fallback=None):
        if x is None:
            return fallback
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    scale_np = _to_np(scale)
    offset_np = _to_np(offset)

    if scale_np is None:
        # fallback: if normalized action, we don't know true scale; use ones so that 'delta' means rad directly
        scale_np = np.ones((action_dim,), dtype=np.float32)
    scale_np = np.asarray(scale_np, dtype=np.float32).reshape(-1)
    if offset_np is None:
        offset_np = np.zeros((action_dim,), dtype=np.float32)
    offset_np = np.asarray(offset_np, dtype=np.float32).reshape(-1)

    # Decide mode
    mode = "delta"  # safer default for [-1,1] in this task (prevents sign/offset mistakes)
    hint = ""
    try:
        # Look for strings like "delta_joint_pos" / "delta" in controller config
        s = (str(type(ctrl)) + " " + repr(getattr(ctrl, "config", "")) + " " + repr(getattr(ctrl, "control_type", ""))).lower()
        if "absolute" in s or "target" in s or "pd_joint_pos" in s or "joint_pos" in s:
            mode = "absolute"
            hint = "string-hint: absolute"
        if "delta" in s:
            mode = "delta"
            hint = "string-hint: delta"
    except Exception:
        pass

    # Heuristic override: small action_scale (<< 1 rad) almost always indicates delta increments
    try:
        sc = np.nanmax(np.abs(scale_np[:dof]))
        if np.isfinite(sc) and sc < 1.0:
            mode = "delta"
            hint = hint or "scale<1rad -> delta"
    except Exception:
        pass

    # Clip scale to avoid divide-by-zero
    scale_np = np.where(np.abs(scale_np) < 1e-6, 1.0, scale_np).astype(np.float32)

    if os.environ.get("S2R_DEBUG", "0") not in ("0", "false", "False", ""):
        print(f"[DEBUG] inferred action mode = {mode} ({hint if hint else 'no-hint'})")
        try:
            print("[DEBUG] controller type:", type(ctrl))
        except Exception:
            pass
        try:
            print("[DEBUG] action_scale(first 7) =", scale_np[:min(7, len(scale_np))])
            print("[DEBUG] action_offset(first 7) =", offset_np[:min(7, len(offset_np))])
        except Exception:
            pass

    return mode, scale_np, offset_np, ctrl


def desired_qpos_to_env_action(desired_qpos_rad: np.ndarray,
                               current_qpos_rad: np.ndarray,
                               action_space,
                               mode: str,
                               action_scale: np.ndarray,
                               action_offset: np.ndarray,
                               dof: int) -> np.ndarray:
    """Convert desired absolute qpos(rad) to env action under inferred transform."""
    desired = np.asarray(desired_qpos_rad, dtype=np.float32).reshape(-1)[:dof]
    current = np.asarray(current_qpos_rad, dtype=np.float32).reshape(-1)[:dof]

    scale = np.asarray(action_scale, dtype=np.float32).reshape(-1)[:dof]
    offset = np.asarray(action_offset, dtype=np.float32).reshape(-1)[:dof]
    scale = np.where(np.abs(scale) < 1e-6, 1.0, scale)

    if mode == "delta":
        # target = current + action * scale + offset
        raw = (desired - current - offset) / scale
    else:
        # target = action * scale + offset
        raw = (desired - offset) / scale

    # Clip by action_space bounds if available
    try:
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)[:dof]
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)[:dof]
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            raw = np.clip(raw, low, high)
    except Exception:
        pass
    return raw.astype(np.float32)
def clamp_qpos_to_limits(qpos_rad: np.ndarray, joint_limits_rad: np.ndarray, dof: int) -> np.ndarray:
    q = np.asarray(qpos_rad, dtype=np.float32).reshape(-1)[:dof]
    jl = _sanitize_joint_limits_rad(joint_limits_rad, dof)
    return np.clip(q, jl[:, 0], jl[:, 1]).astype(np.float32)


def set_sim_qpos_from_values(base_sim_env, values):
    robot = base_sim_env.agent.robot
    sim_qpos = robot.get_qpos().clone()
    target = torch.as_tensor(values, dtype=sim_qpos.dtype, device=sim_qpos.device)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    sim_dof = sim_qpos.shape[-1]
    gripper_val = target[..., -1:].clone()
    if target.shape[-1] < sim_dof and target.shape[-1] > 0:
        pad_len = sim_dof - target.shape[-1]
        if pad_len > 0:
            pad = target[..., -1:].expand(target.shape[0], pad_len)
            target = torch.cat([target, pad], dim=-1)
    if target.shape[-1] > sim_dof:
        target = target[..., :sim_dof]
    drive_idx = getattr(base_sim_env, "_finger_drive_index", None)
    if drive_idx is not None:
        target[..., drive_idx] = gripper_val

    sim_qpos = target.to(sim_qpos.dtype)
    robot.set_qpos(sim_qpos)
    # base_sim_env.step(action)


def sync_real_pose_into_sim(real_env, slider_dim):
    """Read the real robot pose, mirror it into sim, and return padded values for sliders."""
    real_qpos = get_real_qpos_safe(real_env)
    if real_qpos is None:
        raise RuntimeError("Failed to read real robot qpos while syncing.")
    real_qpos = real_qpos.detach().cpu().reshape(-1)
    padded = pad_qpos(real_qpos.numpy(), slider_dim)
    set_sim_qpos_from_values(real_env.base_sim_env, padded)
    return padded


def get_real_qpos_safe(real_env):
    """Helper that returns real qpos tensor or None if the hardware read failed."""
    try:
        return real_env.agent.robot.get_qpos()
    except Exception as exc:
        print(f"[sync warning] Failed to read real qpos: {exc}")
        return None


S2R_DEBUG = os.environ.get("S2R_DEBUG", "1").strip() not in ("0", "false", "False", "FALSE", "")
GRIPPER_DRIVE_JOINT = "gripper_Right_1_Joint"
GRIPPER_JOINT_NAMES = [
    "gripper_Right_1_Joint",
    "gripper_Right_Support_Joint",
    "gripper_Right_2_Joint",
    "gripper_Left_1_Joint",
    "gripper_Left_Support_Joint",
    "gripper_Left_2_Joint",
]
GRIPPER_LIMIT_OVERRIDE = (0.0, 0.91)  # radians, from URDF

class ActionTuner:
    def __init__(self, org_pos, dim, gripper_index=None, limits_deg=None, limits_rad=None):
        self.dim = dim
        self.gripper_index = gripper_index
        self.limits_deg = limits_deg  # list[(min_deg,max_deg)] or None
        self.limits_rad = limits_rad

        # Avoid creating a second Tk root if one already exists (e.g., matplotlib TkAgg).
        master = tk._default_root
        if master is None:
            self.root = tk.Tk()
        else:
            self.root = tk.Toplevel(master)
        self.root.title("Action Tuner")

        self.v = [0.0] * self.dim
        self._suspend_updates = False
        self.updated = True
        self._is_dragging = False
        self._last_real_readback = None

        self._last_gui_print_t = 0.0

        # Initial values
        self.vals = []
        self.display_texts = []
        for i in range(self.dim):
            init_val = org_pos[i] if i < len(org_pos) else 0.0
            if self.gripper_index is not None and i == self.gripper_index:
                lo, hi = self._get_limits_rad(i)
                span = max(hi - lo, 1e-6)
                norm_val = (init_val - lo) / span
                self.vals.append(tk.DoubleVar(value=float(np.clip(norm_val, 0.0, 1.0))))
            else:
                self.vals.append(tk.DoubleVar(value=float(np.rad2deg(init_val))))

        # Sliders
        for i in range(self.dim):
            slider_name = "Gripper" if (self.gripper_index is not None and i == self.gripper_index) else f"Joint {i}"
            if self.gripper_index is not None and i == self.gripper_index:
                mn, mx = (0.0, 1.0)
            elif self.limits_deg is not None and i < len(self.limits_deg):
                mn, mx = self.limits_deg[i]
            else:
                mn, mx = (-180.0, 180.0)
            self.create_slider(slider_name, self.vals[i], mn, mx, i)

        # Display label
        self.label = tk.Label(self.root, text="")
        self.update_label()
        self.label.grid(row=self.dim, column=0, columnspan=2, pady=10)

        # Bind update events
        for i in range(self.dim):
            self.vals[i].trace("w", self.update_label)
        self.update_readback()

    def create_slider(self, name, variable, min_val, max_val, row):
        """Create a slider."""
        label = tk.Label(self.root, text=f"{name}")
        label.grid(row=row, column=0, padx=10, pady=5)

        slider = ttk.Scale(self.root, from_=min_val, to=max_val,
                           orient="horizontal", variable=variable,
                           command=self.update_label, length=1000)
        slider.grid(row=row, column=1, padx=10, pady=5)
        slider.bind("<ButtonPress-1>", self._on_drag_start)
        slider.bind("<ButtonRelease-1>", self._on_drag_end)

        info_var = tk.StringVar(value="")
        info_label = tk.Label(self.root, textvariable=info_var, anchor="w")
        info_label.grid(row=row, column=2, padx=10, pady=5, sticky="w")
        self.display_texts.append(info_var)

    def update_label(self, *args):
        """Update internal action vector (stored in radians)."""
        v = []
        for i in range(self.dim):
            val = self.vals[i].get()
            if self.gripper_index is not None and i == self.gripper_index:
                lo, hi = self._get_limits_rad(i)
                span = max(hi - lo, 1e-6)
                v.append(float(lo + val * span))
            else:
                v.append(float(np.deg2rad(val)))
        self.v = v

        if not self._suspend_updates:
            self.updated = True
            if S2R_DEBUG:
                now = time.monotonic()
                # throttle prints
                if now - self._last_gui_print_t > 0.25:
                    self._last_gui_print_t = now
                    print("[GUI] updated (deg) joint0=", self.vals[0].get())
        self.update_readback()

    def consume_updated_flag(self):
        if self.updated:
            self.updated = False
            return True
        return False

    def set_values(self, values):
        """Programmatically update slider values without triggering a push."""
        self._suspend_updates = True
        padded = list(values)[: self.dim]
        if len(padded) < self.dim:
            padded.extend([0.0] * (self.dim - len(padded)))

        for i in range(self.dim):
            if self.gripper_index is not None and i == self.gripper_index:
                lo, hi = self._get_limits_rad(i)
                span = max(hi - lo, 1e-6)
                norm_val = (float(padded[i]) - lo) / span
                self.vals[i].set(float(np.clip(norm_val, 0.0, 1.0)))
            else:
                self.vals[i].set(float(np.rad2deg(padded[i])))

        self._suspend_updates = False
        self.v = padded
        self.updated = False
        self.update_readback()

    def _on_drag_start(self, _event):
        self._is_dragging = True

    def _on_drag_end(self, _event):
        self._is_dragging = False

    def is_dragging(self) -> bool:
        return self._is_dragging

    def update_readback(self, real_values=None):
        """Update per-joint text showing target vs. actual values."""
        if real_values is not None:
            arr = np.asarray(real_values, dtype=np.float32).reshape(-1)
            self._last_real_readback = arr
        else:
            arr = self._last_real_readback

        for i in range(self.dim):
            if self.gripper_index is not None and i == self.gripper_index:
                lo, hi = self._get_limits_rad(i)
                span = max(hi - lo, 1e-6)
                slider_norm = float(self.vals[i].get())
                desired_rad = lo + slider_norm * span
                desired_deg = float(np.rad2deg(desired_rad))
                actual_rad = float(arr[i]) if arr is not None and i < len(arr) else None
                actual_deg = float(np.rad2deg(actual_rad)) if actual_rad is not None else None
                target_text = f"目标 {desired_deg:.2f}°"
                actual_text = f"实际 {actual_deg:.2f}°" if actual_deg is not None else "实际 --"
            else:
                desired_deg = float(self.vals[i].get())
                actual_deg = float(np.rad2deg(arr[i])) if arr is not None and i < len(arr) else None
                target_text = f"目标 {desired_deg:.2f}°"
                actual_text = f"实际 {actual_deg:.2f}°" if actual_deg is not None else "实际 --"
            if i < len(self.display_texts):
                self.display_texts[i].set(f"{target_text} | {actual_text}")

    def _get_limits_rad(self, idx):
        if self.limits_rad is not None and idx < len(self.limits_rad):
            return self.limits_rad[idx]
        if self.limits_deg is not None and idx < len(self.limits_deg):
            lo_deg, hi_deg = self.limits_deg[idx]
            return np.deg2rad(lo_deg), np.deg2rad(hi_deg)
        return (-np.pi, np.pi)


class ActionTunerRunner:
    """Keeps the Tkinter GUI on the main thread while providing a polling API."""

    def __init__(self, org_pos, dim, gripper_index=None, limits_deg=None, limits_rad=None):
        self.app = ActionTuner(
            org_pos,
            dim,
            gripper_index=gripper_index,
            limits_deg=limits_deg,
            limits_rad=limits_rad,
        )
        self._closed = False

    def pump_events(self):
        """Process Tk events; returns False if the window has been closed."""
        if self._closed:
            return False
        try:
            self.app.root.update_idletasks()
            self.app.root.update()
            return True
        except tk.TclError:
            self._closed = True
            return False
def main(args: Args):
    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="human",
        reward_mode="none",
        render_backend="cpu",
        # High resolution eases manual alignment; training can downscale later.
        sensor_configs=dict(width=1280, height=1280),
        domain_randomization_config=dict(
            max_camera_offset=[0.0, 0.0, 0.0],
            camera_target_noise=0.0,
            camera_view_rot_noise=0.0,
            camera_fov_noise=0.0,
        ),
    )

    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r", encoding="utf-8") as f:
            env_kwargs.update(json.load(f))
        print("Loaded env_kwargs from JSON:", env_kwargs)
    # JSON 里如果带 domain_randomization_config，会覆盖我们在此脚本中用于固定相机的配置。
    # 这里再强制写回一次，避免摄像头继续抖动。
    env_kwargs.setdefault("domain_randomization_config", {})
    env_kwargs["domain_randomization_config"].update(
        max_camera_offset=[0.0, 0.0, 0.0],
        camera_target_noise=0.0,
        camera_view_rot_noise=0.0,
        camera_fov_noise=0.0,
    )
    print(
        "Forced domain_randomization_config for alignment:",
        env_kwargs["domain_randomization_config"],
    )

    sim_env = FlattenRGBDObservationWrapper(gym.make(args.env_id, **env_kwargs))
    real_robot, _ = create_real_robot()
    real_agent = ClippedLeRobotRealAgent(real_robot, use_cached_qpos=False)
    real_env = Sim2RealEnv(
        sim_env=sim_env,
        agent=real_agent,
        real_reset_function=lambda self, seed=None, options=None: self.sim_env.reset(seed=seed, options=options),
    )
    setup_safe_exit(sim_env, real_env, real_agent)
    real_env.reset()

    sim_robot = real_env.base_sim_env.agent.robot
    finger_joint_indices = []
    finger_drive_index = None
    for name in GRIPPER_JOINT_NAMES:
        joint = sim_robot.active_joints_map.get(name)
        if joint is None or joint.active_index is None:
            continue
        idx = int(joint.active_index[0].item())
        finger_joint_indices.append(idx)
        if name == GRIPPER_DRIVE_JOINT:
            finger_drive_index = idx
    if finger_joint_indices:
        setattr(real_env.base_sim_env, "_finger_joint_indices", finger_joint_indices)
        setattr(
            real_env.base_sim_env,
            "_finger_drive_index",
            finger_drive_index if finger_drive_index is not None else finger_joint_indices[0],
        )
        real_agent._gripper_drive_index = finger_drive_index if finger_drive_index is not None else finger_joint_indices[0]

    action_dim = int(np.prod(real_env.action_space.shape))
    has_gripper_action = finger_drive_index is not None and action_dim > 0
    arm_dof = action_dim - 1 if has_gripper_action else action_dim
    include_gripper = args.enable_gripper_control and has_gripper_action
    slider_dim = arm_dof + (1 if include_gripper else 0)
    gripper_index = arm_dof if include_gripper else None

    if args.auto_sync_on_start:
        try:
            initial_pose = sync_real_pose_into_sim(real_env, slider_dim)
        except RuntimeError:
            print("无法读取真实机械臂姿态，退出。")
            return
    else:
        real_pose_tensor = get_real_qpos_safe(real_env)
        if real_pose_tensor is not None:
            initial_pose = pad_qpos(real_pose_tensor.detach().cpu().numpy(), slider_dim)
        else:
            sim_qpos = real_env.base_sim_env.agent.robot.get_qpos()
            initial_pose = pad_qpos(sim_qpos.detach().cpu().numpy(), slider_dim)

    base_pose = initial_pose[:arm_dof]
    if include_gripper:
        sim_qpos_full = real_env.base_sim_env.agent.robot.get_qpos().detach().cpu().flatten().numpy()
        if finger_drive_index is not None and finger_drive_index < len(sim_qpos_full):
            gripper_default = float(sim_qpos_full[finger_drive_index])
            lo_override, hi_override = GRIPPER_LIMIT_OVERRIDE
            gripper_default = float(np.clip(gripper_default, lo_override, hi_override))
        else:
            gripper_default = 0.0
        base_pose = np.concatenate([base_pose, [gripper_default]])
    initial_pose = np.asarray(base_pose, dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if hasattr(fig.canvas.manager, "key_press_handler_id"):
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.manager.key_press_handler_id = None

    axes[0].set_title("Real Camera")
    axes[1].set_title("Sim Camera")
    axes[2].set_title("Overlay")

    overlaid_imgs, sim_imgs, real_imgs = overlay_envs(sim_env, real_env)
    im_real = axes[0].imshow(real_imgs)
    im_sim = axes[1].imshow(sim_imgs)
    im_overlay = axes[2].imshow(overlaid_imgs)

    def refresh_overlay():
        overlaid, sim_only, real_only = overlay_envs(sim_env, real_env)
        im_real.set_data(real_only)
        im_sim.set_data(sim_only)
        im_overlay.set_data(overlaid)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    refresh_overlay()
    plt.show(block=False)
    print("Close the camera alignment window once satisfied.")

    # Joint limits from sim (rad), used for slider ranges + safety clamping + normalized action mapping.
    try:
        qlim = real_env.base_sim_env.agent.robot.get_qlimits()
        if isinstance(qlim, torch.Tensor):
            qlim_np = qlim.detach().cpu().numpy()
        else:
            qlim_np = np.asarray(qlim)
        # expected shape: (1, dof, 2) or (dof,2)
        if qlim_np.ndim == 3:
            full_joint_limits_rad = qlim_np[0]
        else:
            full_joint_limits_rad = qlim_np
        joint_limits_arm = full_joint_limits_rad[:arm_dof]
        if include_gripper and finger_drive_index is not None and finger_drive_index < len(full_joint_limits_rad):
            gripper_limit = full_joint_limits_rad[finger_drive_index][None, :]
            joint_limits_rad = np.concatenate([joint_limits_arm, gripper_limit], axis=0)
        else:
            joint_limits_rad = joint_limits_arm
    except Exception as exc:
        print("[WARN] Failed to read sim joint limits; fallback to [-pi,pi]. Error:", repr(exc))
        joint_limits_rad = np.stack([[-np.pi, np.pi]] * slider_dim, axis=0).astype(np.float32)
        full_joint_limits_rad = joint_limits_rad

    if include_gripper and slider_dim > arm_dof:
        lo_override, hi_override = GRIPPER_LIMIT_OVERRIDE
        if joint_limits_rad.shape[0] >= slider_dim:
            joint_limits_rad[-1, 0] = lo_override
            joint_limits_rad[-1, 1] = hi_override
        if (
            finger_drive_index is not None
            and full_joint_limits_rad is not None
            and finger_drive_index < len(full_joint_limits_rad)
        ):
            full_joint_limits_rad[finger_drive_index, 0] = lo_override
            full_joint_limits_rad[finger_drive_index, 1] = hi_override

    gripper_drive_limit = None
    if include_gripper:
        lo_override, hi_override = GRIPPER_LIMIT_OVERRIDE
        joint_limits_rad[-1] = np.array([lo_override, hi_override], dtype=np.float32)
        if finger_drive_index is not None and finger_drive_index < len(full_joint_limits_rad):
            full_joint_limits_rad[finger_drive_index] = joint_limits_rad[-1]
        if S2R_DEBUG:
            print(f"[DEBUG] Override gripper limits to {joint_limits_rad[-1].tolist()} rad")
        drive_limit = joint_limits_rad[-1]
        real_agent.set_gripper_limits(drive_limit)
        gripper_drive_limit = drive_limit

    # Slider limits (degrees for display + radians for conversions)
    limits_deg = []
    limits_rad = []
    for i in range(slider_dim):
        lo, hi = float(joint_limits_rad[i, 0]), float(joint_limits_rad[i, 1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
            lo, hi = -np.pi, np.pi
        limits_rad.append((lo, hi))
        limits_deg.append((float(np.rad2deg(lo)), float(np.rad2deg(hi))))

    ACTION_NORMALIZED = _is_normalized_action_space(real_env.action_space, slider_dim)

    # Debug: print key ranges once
    if S2R_DEBUG:
        try:
            low = np.asarray(real_env.action_space.low, dtype=np.float32).reshape(-1)[:slider_dim]
            high = np.asarray(real_env.action_space.high, dtype=np.float32).reshape(-1)[:slider_dim]
            print("[DEBUG] action_space first-dof low/high:", float(low.min()), float(low.max()), float(high.min()), float(high.max()))
        except Exception as exc:
            print("[DEBUG] action_space bounds unavailable:", repr(exc))
        print("[DEBUG] ACTION_NORMALIZED =", ACTION_NORMALIZED)
        print("[DEBUG] joint_limits_deg (first 7) =", limits_deg[:min(7, len(limits_deg))])

    # Infer controller action transform (delta vs absolute) to avoid sign/offset mismatches.
    action_mode, action_scale, action_offset, _ctrl_obj = infer_action_transform(
        real_env, real_env.action_space, action_dim=action_dim, dof=slider_dim
    )
    if S2R_DEBUG:
        print('[DEBUG] action_mode =', action_mode)

    actiontuner = ActionTunerRunner(
        initial_pose.tolist(),
        slider_dim,
        gripper_index=gripper_index,
        limits_deg=limits_deg,
        limits_rad=limits_rad,
    )
    actiontuner.app.updated = False

    # Cache of last commanded pose (qpos rad)
    prev_slider = np.array(initial_pose[:slider_dim], dtype=np.float32)
    pending_target = prev_slider.copy()
    latest_readback = prev_slider.copy()
    max_step_rad = np.deg2rad(args.max_step_deg)
    sync_tol_rad = np.deg2rad(args.sync_tol_deg)

    CONTROL_HZ = 50.0
    VIS_HZ = 5.0
    next_ctrl_t = time.perf_counter()
    last_vis_t = 0.0
    last_read_t = 0.0

    while True:
        start_time=time.time()
        now = time.perf_counter()
        # --- 控制频率锁定到 50Hz ---
        if now < next_ctrl_t:
            time.sleep(max(0.0, next_ctrl_t - now))
            continue
        next_ctrl_t += 1.0 / CONTROL_HZ

        if not actiontuner.pump_events():
            break

        if not actiontuner.pump_events():
            print("Action tuner window closed, exiting control loop.")
            break

        # 1) 读取滑块目标（pending_target 是“希望 sim 达到的 qpos”）
        action = np.array(actiontuner.app.v[:slider_dim], dtype=np.float32)
        slider_changed = actiontuner.app.consume_updated_flag()
        if not slider_changed:
            slider_changed = not np.allclose(action, prev_slider, atol=1e-4)

        if slider_changed:
            target_qpos = clamp_qpos_to_limits(action, joint_limits_rad, slider_dim)
            prev_slider = target_qpos.copy()
            pending_target = target_qpos.copy()

        desired_gripper_norm = None
        if include_gripper and gripper_drive_limit is not None and gripper_index is not None:
            lo, hi = float(gripper_drive_limit[0]), float(gripper_drive_limit[1])
            span = max(hi - lo, 1e-6)
            desired_gripper_norm = float(np.clip((pending_target[gripper_index] - lo) / span, 0.0, 1.0))
        print("time1",time.time()-start_time)

        # 2) 当前 sim 状态（用 sim 做“主状态”）
        sim_qpos_full = real_env.base_sim_env.agent.robot.get_qpos().detach().cpu().flatten().numpy()

        sim_state = np.zeros((slider_dim,), dtype=np.float32)
        sim_state[:arm_dof] = sim_qpos_full[:arm_dof]
        if include_gripper:
            if finger_drive_index is not None and finger_drive_index < sim_qpos_full.shape[0]:
                sim_state[arm_dof] = float(sim_qpos_full[finger_drive_index])
            else:
                sim_state[arm_dof] = float(sim_qpos_full[-1])
        print("time2", time.time() - start_time)

        # 3) 让 sim 追踪 pending_target（把“目标 qpos”转换成 env action，然后只 step 一次 sim）
        delta = pending_target - sim_state
        max_err = float(np.max(np.abs(delta)))
        if max_err > sync_tol_rad:
            clipped_delta = np.clip(delta, -max_step_rad, max_step_rad)
            commanded = sim_state + clipped_delta

            env_action = desired_qpos_to_env_action(
                commanded,
                sim_state,
                real_env.action_space,   # action_space 用哪个都行，只要跟 sim_env 一致；你这里 real_env.action_space = sim_env.action_space
                action_mode,
                action_scale,
                action_offset,
                slider_dim,
            )
            if include_gripper and gripper_drive_limit is not None and gripper_index is not None:
                lo, hi = float(gripper_drive_limit[0]), float(gripper_drive_limit[1])
                span = max(hi - lo, 1e-6)
                norm = np.clip((commanded[gripper_index] - lo) / span, 0.0, 1.0)
                env_action[gripper_index] = norm * 2.0 - 1.0
            env_action_padded = pad_qpos(env_action, action_dim)

            commanded_full = sim_qpos_full.copy()
            commanded_full[:arm_dof] = commanded[:arm_dof]
            if include_gripper and gripper_index is not None:
                if finger_drive_index is not None and finger_drive_index < commanded_full.shape[0]:
                    commanded_full[finger_drive_index] = commanded[gripper_index]
                else:
                    commanded_full[-1] = commanded[gripper_index]

            try:
                # print("env_action_padded:",env_action_padded)
                sim_env.step(env_action_padded)   # ✅ 只 step 一次 sim
                print("time3。1", time.time() - start_time)
            except Exception as exc:
                print("[WARN] sim_env.step failed", exc)

            try:
                real_agent.apply_sim_qpos(commanded_full, override_gripper_norm=desired_gripper_norm)
                print("time3。2", time.time() - start_time)
            except Exception as exc:
                print("[WARN] real_agent.apply_sim_qpos (commanded) failed:", exc)

        else:
            time.sleep(0.005)
        print("time3", time.time() - start_time)

        # 4) step 完后的 sim qpos => 作为 real 的跟随目标（real 跟随 sim）
        sim_qpos_full = real_env.base_sim_env.agent.robot.get_qpos().detach().cpu().flatten().numpy()

        if S2R_DEBUG and include_gripper and desired_gripper_norm is not None:
            if finger_drive_index is not None and finger_drive_index < sim_qpos_full.shape[0]:
                sim_drive_val = float(sim_qpos_full[finger_drive_index])
            else:
                sim_drive_val = float(sim_qpos_full[-1])
            approx_counts = int(round(desired_gripper_norm * 9000.0))
            # print(f"[DEBUG] gripper sim={sim_drive_val:.4f} rad  slider_norm={desired_gripper_norm:.3f}  -> cmd≈{approx_counts}")

        # try:
            # 推荐用你类里已经写好的 apply_sim_qpos：它会同时处理 arm + gripper 的映射/归一化
            # real_agent.apply_sim_qpos(sim_qpos_full, override_gripper_norm=desired_gripper_norm)
        # except Exception as exc:
        #     print("[WARN] real_agent.apply_sim_qpos failed:", exc)
        print("time4", time.time() - start_time)

        # 5) GUI 显示：真实读回（显示 real 实际值，不再反向覆盖 sim）
        READBACK_HZ = 5.0
        if now - last_read_t >= 1.0 / READBACK_HZ:
            last_read_t = now
            real_state_tensor = get_real_qpos_safe(real_env)
            if real_state_tensor is not None:
                real_state = real_state_tensor.detach().cpu().flatten().numpy()
                real_state_padded = pad_qpos(real_state, slider_dim)
                actiontuner.app.update_readback(real_state_padded)
            print("time5", time.time() - start_time)

        # 6) 渲染/overlay
        need_vis = slider_changed or (max_err > sync_tol_rad)
        if need_vis and (now - last_vis_t >= 1.0 / VIS_HZ):
            last_vis_t = now
            sim_env.render()
            refresh_overlay()
        print("time6", time.time() - start_time)

if __name__ == "__main__":
    args = tyro.cli(Args)
    print("env_id =", args.env_id)
    main(args)
