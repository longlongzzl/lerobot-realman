"""
Interactive offline attention viewer.
Reads rgb + real_qpos_deg.npy and runs the policy to visualize attention maps.
"""
import os
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tyro
import imageio
import matplotlib.pyplot as plt

from lerobot_sim2real.rl.ppo_rgb import Agent


class _DummySpace:
    def __init__(self, shape):
        self.shape = tuple(shape)


class _DummyEnv:
    def __init__(self, action_dim: int):
        self.single_action_space = _DummySpace((action_dim,))
        self.unwrapped = self


@dataclass
class Args:
    checkpoints: List[str] = field(default_factory=list)
    """Policy checkpoints (state_dict). Pass multiple paths to compare attentions."""
    checkpoint: List[str] = field(default_factory=list)
    """Policy checkpoints (legacy flag). Can pass multiple paths."""
    checkpoint_dir: Optional[str] = None
    """Directory containing checkpoints. If set, all .pt/.pth/.ckpt in this folder are used."""
    input_dir: Optional[str] = None
    """Run root (contains episode_*/step_*) or a single step dir with real_qpos_deg.npy. If None, a folder picker is shown."""
    device: Optional[str] = None
    """cuda/cpu. If None, auto."""
    net_type: Optional[str] = None
    """Override policy net type. If None, infer from checkpoint."""
    choice: int = 10
    tcl_choice: int = 5
    act_type: int = 1
    z_detach: bool = False
    sigma: float = 0.2
    search_entropy: float = 0.0
    critic_use_pi: Optional[bool] = None
    state_dim: Optional[int] = None
    """Override policy state dim. If None, infer from each checkpoint."""
    qpos_mode: str = "deg"
    """qpos input unit: deg or rad"""
    start_index: int = 0
    """Start index for browsing."""


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "agent"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")
    return _normalize_state_dict(ckpt)


def _infer_state_dim(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    for k, v in state_dict.items():
        if k.endswith("feature_net.extractors.state.0.weight") and v.ndim == 2:
            return int(v.shape[1])
    return None


def _infer_action_dim(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    if "actor_logstd" in state_dict:
        return int(state_dict["actor_logstd"].shape[-1])
    cand = []
    for k, v in state_dict.items():
        if "actor_mean" in k and k.endswith("weight") and v.ndim == 2:
            cand.append(int(v.shape[0]))
    if cand:
        return min(cand)
    return None


def _infer_private_dim(state_dict: Dict[str, torch.Tensor], state_dim: Optional[int]) -> int:
    if state_dim is None:
        return 0
    for k, v in state_dict.items():
        if k.endswith("critic_representation.0.weight") and v.ndim == 2:
            return int(v.shape[1] - state_dim)
    return 0


def _infer_net_type(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    for k in state_dict:
        if k.endswith("feature_net.extractors.rgb_0.attn_head.weight"):
            return "3"
    return None


def _list_step_dirs(input_dir: str) -> Tuple[List[str], bool]:
    step_marker = os.path.join(input_dir, "real_qpos_deg.npy")
    if os.path.isfile(step_marker):
        return [input_dir], True
    step_dirs = []
    for root, dirs, files in os.walk(input_dir):
        if "real_qpos_deg.npy" in files:
            step_dirs.append(root)
    step_dirs.sort()
    return step_dirs, False


def _pick_dir(title: str) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title)
        root.destroy()
        if path:
            return path
    except Exception as exc:
        print(f"[Picker] failed to open folder dialog: {exc}")
    return None


def _pick_input_dir() -> Optional[str]:
    return _pick_dir("Select debug_steps folder")


def _pick_checkpoint_dir() -> Optional[str]:
    return _pick_dir("Select checkpoint folder")


def _list_checkpoints_in_dir(folder: str) -> List[str]:
    exts = (".pt", ".pth", ".ckpt")
    items = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.endswith(exts):
            items.append(path)
    return items


def _prepare_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (3, 4):
        img = img.transpose(1, 2, 0)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.max() <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _load_rgb(step_dir: str, key: str = "rgb_0") -> Optional[np.ndarray]:
    npy_path = os.path.join(step_dir, f"{key}.npy")
    png_path = os.path.join(step_dir, f"{key}.png")
    if os.path.isfile(npy_path):
        img = np.load(npy_path)
    elif os.path.isfile(png_path):
        img = imageio.imread(png_path)
    else:
        return None
    return _prepare_rgb(img)


def _build_state(qpos_deg: np.ndarray, state_dim: int, qpos_mode: str) -> np.ndarray:
    qpos = qpos_deg.astype(np.float32)
    if qpos_mode == "rad":
        qpos = np.deg2rad(qpos)
    state = np.zeros((state_dim,), dtype=np.float32)
    n = min(state_dim, qpos.shape[0])
    state[:n] = qpos[:n]
    return state


def _upsample_attention(attn_map: torch.Tensor, target_h: int, target_w: int) -> np.ndarray:
    if isinstance(attn_map, np.ndarray):
        attn_map = torch.from_numpy(attn_map)
    if attn_map.ndim == 2:
        attn_map = attn_map.unsqueeze(0).unsqueeze(0)
    elif attn_map.ndim == 3:
        if attn_map.shape[0] == 1:
            attn_map = attn_map.unsqueeze(0)
        else:
            attn_map = attn_map.unsqueeze(1)
    attn_upsampled = F.interpolate(
        attn_map, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    return attn_upsampled[0, 0].detach().cpu().numpy()


def _filter_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            filtered[k] = v
    return filtered


def _has_rgb(step_dir: str) -> bool:
    return (
        os.path.isfile(os.path.join(step_dir, "rgb_0.npy"))
        or os.path.isfile(os.path.join(step_dir, "rgb_0.png"))
    )


def _build_agent(
    checkpoint: str,
    rgb_shape: Tuple[int, int, int],
    device: torch.device,
    args: Args,
):
    state_dict = _load_checkpoint(checkpoint)
    net_type = args.net_type or _infer_net_type(state_dict) or "3"

    state_dim = args.state_dim or _infer_state_dim(state_dict)
    if state_dim is None:
        raise ValueError(f"Failed to infer state_dim for {checkpoint}. Provide --state-dim.")

    action_dim = _infer_action_dim(state_dict)
    if action_dim is None:
        raise ValueError(f"Failed to infer action_dim for {checkpoint}.")

    private_dim = _infer_private_dim(state_dict, state_dim)
    if args.critic_use_pi is None:
        critic_use_pi = any(k.startswith("critic_representation.") for k in state_dict)
    else:
        critic_use_pi = args.critic_use_pi

    h, w, c = rgb_shape
    sample_obs = {
        "rgb_0": torch.zeros((1, h, w, c), dtype=torch.uint8),
        "state": torch.zeros((1, state_dim), dtype=torch.float32),
    }
    dummy_env = _DummyEnv(action_dim)
    agent = Agent(
        None,
        args.sigma,
        dummy_env,
        sample_obs=sample_obs,
        net_type=net_type,
        choice=args.choice,
        tcl_choice=args.tcl_choice,
        device=device,
        z_detach=args.z_detach,
        private_dim=private_dim,
        search_entropy=args.search_entropy,
        act_type=args.act_type,
        critic_use_pi=critic_use_pi,
    )
    filtered = _filter_state_dict(agent, state_dict)
    agent.load_state_dict(filtered, strict=False)
    agent.eval()
    agent.to(device)
    return agent, {
        "checkpoint": checkpoint,
        "net_type": net_type,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "private_dim": private_dim,
        "critic_use_pi": critic_use_pi,
    }


def main(args: Args) -> None:
    if args.qpos_mode not in ("deg", "rad"):
        raise ValueError(f"Unsupported qpos_mode: {args.qpos_mode}")
    ckpt_list = []
    if args.checkpoints:
        ckpt_list.extend(args.checkpoints)
    if args.checkpoint:
        ckpt_list.extend(args.checkpoint)

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir and os.path.isdir(ckpt_dir):
        ckpt_list.extend(_list_checkpoints_in_dir(ckpt_dir))

    if not ckpt_list:
        ckpt_dir = _pick_checkpoint_dir()
        if ckpt_dir and os.path.isdir(ckpt_dir):
            ckpt_list.extend(_list_checkpoints_in_dir(ckpt_dir))

    if not ckpt_list:
        raise ValueError("No checkpoints provided. Use --checkpoints/--checkpoint or select a checkpoint folder.")

    seen = set()
    deduped = []
    for p in ckpt_list:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    args.checkpoints = deduped
    input_dir = args.input_dir
    if input_dir is None or not os.path.isdir(input_dir):
        input_dir = _pick_input_dir()
    if input_dir is None or not os.path.isdir(input_dir):
        raise ValueError("No valid input_dir selected.")

    step_dirs, _ = _list_step_dirs(input_dir)
    step_dirs = [
        d for d in step_dirs
        if os.path.isfile(os.path.join(d, "real_qpos_deg.npy")) and _has_rgb(d)
    ]
    if not step_dirs:
        raise ValueError(f"No step dirs found under: {input_dir}")

    rgb0 = _load_rgb(step_dirs[0], "rgb_0")
    if rgb0 is None:
        raise ValueError(f"Missing rgb_0 in {step_dirs[0]}")
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = rgb0.shape[:2]
    rgb_shape = (h, w, 3)

    agents = []
    agent_metas = []
    for ckpt in args.checkpoints:
        agent, meta = _build_agent(ckpt, rgb_shape, device, args)
        agents.append(agent)
        agent_metas.append(meta)

    total = len(step_dirs)
    idx = max(0, min(args.start_index, total - 1))
    cache = {}
    n_policies = len(agents)
    total_panels = n_policies + 1
    n_cols = min(5, total_panels)
    n_rows = int(math.ceil(total_panels / n_cols))
    fig_w = max(6, 4 * n_cols)
    fig_h = max(4, 3.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_policies == 0:
        raise ValueError("No valid agents constructed.")
    if isinstance(axes, np.ndarray):
        axes = list(axes.ravel())
    else:
        axes = [axes]
    ax_rgb = axes[0]
    ax_rgb.axis("off")
    ax_rgb.set_title("RGB")
    im_rgb = ax_rgb.imshow(rgb0)

    attn_axes = axes[1:total_panels]
    im_attns = []
    for i, ax in enumerate(attn_axes):
        ax.axis("off")
        ckpt_name = os.path.basename(args.checkpoints[i])
        ax.set_title(f"Attn {i + 1}: {ckpt_name}")
        im_attns.append(ax.imshow(np.zeros((rgb0.shape[0], rgb0.shape[1])), cmap="jet", vmin=0, vmax=1))
    for ax in axes[total_panels:]:
        ax.axis("off")

    def compute_item(i: int):
        if i in cache:
            return cache[i]
        step_dir = step_dirs[i]
        rgb = _load_rgb(step_dir, "rgb_0")
        if rgb is None:
            raise ValueError(f"Missing rgb_0 in {step_dir}")
        qpos_path = os.path.join(step_dir, "real_qpos_deg.npy")
        qpos_deg = np.load(qpos_path)
        rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).to(device)
        attn_list = []
        for agent, meta in zip(agents, agent_metas):
            state = _build_state(qpos_deg, meta["state_dim"], args.qpos_mode)
            obs = {
                "rgb_0": rgb_tensor,
                "state": torch.from_numpy(state).unsqueeze(0).to(device),
            }
            with torch.no_grad():
                _ = agent.get_action(obs, None, deterministic=True)
            attn_maps = agent.feature_net.last_attn_maps or {}
            if "rgb_0" not in attn_maps:
                attn = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
            else:
                attn = _upsample_attention(attn_maps["rgb_0"], rgb.shape[0], rgb.shape[1])
            attn_list.append(attn)
        cache[i] = (rgb, attn_list, step_dir)
        return cache[i]

    def update():
        nonlocal idx
        idx = max(0, min(idx, total - 1))
        rgb, attn_list, step_dir = compute_item(idx)
        im_rgb.set_data(rgb)
        for im, attn in zip(im_attns, attn_list):
            im.set_data(attn)
        rel = os.path.relpath(step_dir, input_dir)
        fig.suptitle(f"{idx + 1}/{total}  {rel}  (up/down to switch, q to quit)")
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx
        if event.key == "up":
            idx = min(total - 1, idx + 1)
            update()
        elif event.key == "down":
            idx = max(0, idx - 1)
            update()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
