import os
from typing import Dict, List, Union
from pathlib import Path

import cv2
import torch
import sapien
from PIL import Image

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import SimConfig

import math, random, cv2, torch, torch.nn as nn
import kornia.augmentation as Kaug
from kornia.augmentation.container import ImageSequential
import matplotlib.pyplot as plt
from kornia.augmentation._2d.base import AugmentationBase2D


def _to_same_device(d, dev):
    """把随机参数 dict 全部迁移到输入 dev；非 Tensor 原样返回。"""
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}


def _rand_like(shape, x):  # 统一 dtype / device / layout
    return torch.rand(*shape, device=x.device, dtype=x.dtype)


def _randint_like(low, high, shape, x, dtype=None):
    return torch.randint(low, high, shape, device=x.device,
                         dtype=dtype or x.dtype)


# -----------------------------------------------------------
# 1) Sun‑Flare
# -----------------------------------------------------------
# class RandomSunFlareK(AugmentationBase2D):
#     def __init__(self, src_radius=50, angle_range=(0.0, 0.25),
#                  alpha=(.4, 1.0), gamma=(1.5, 3.0),
#                  p=0.7, same_on_batch=False, keepdim=True):
#         super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
#         self.src_radius, self.angle_range = src_radius, angle_range
#         self.alpha_range, self.gamma_range = alpha, gamma
#         self.register_buffer("_grid", None, persistent=False)
#
#     # ---------- 采参 ----------
#     def generate_parameters(self, batch_shape):
#         B, _, H, W = batch_shape
#         dev = self.device  # Kornia 在 .to() 时会同步更新
#         theta = torch.empty(B, device=dev).uniform_(
#             self.angle_range[0]*2*math.pi, self.angle_range[1]*2*math.pi)
#         return dict(
#             cx=(W*torch.cos(theta)).round(),
#             cy=(H*torch.sin(theta)).round(),
#             radius=torch.randint(self.src_radius//2, self.src_radius+1, (B,), device=dev),
#             alpha=torch.empty(B, device=dev).uniform_(*self.alpha_range),
#             gamma=torch.empty(B, device=dev).uniform_(*self.gamma_range)
#         )
#
#     # ---------- 变换 ----------
#     def apply_transform(self, x, p, flags=None, transform=None):
#         p = _to_same_device(p, x.device)
#         B, _, H, W = x.shape
#         if (self._grid is None) or self._grid.shape[-2:] != (H, W):
#             yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
#                                     torch.arange(W, device=x.device), indexing="ij")
#             self._grid = torch.stack((yy, xx), 0)
#         yy, xx = self._grid
#
#         mask = 1 - (((xx[None]-p["cx"][:,None,None])**2 +
#                      (yy[None]-p["cy"][:,None,None])**2).sqrt_()
#                     / p["radius"][:,None,None]).clamp_(0, 1)
#         mask = mask.pow_(p["gamma"][:,None,None]) * p["alpha"][:,None,None]
#         return (x + mask[:, None]).clamp_(0, 1)


class RandomSunFlareK(AugmentationBase2D):
    def __init__(self, src_radius=50, angle_range=(0.0, 0.25),
                 alpha=(.4, 1.0), gamma=(1.5, 3.0),
                 color_range=(0.0, 1.0),  # 随机RGB范围
                 max_flares=3,  # 最多几个光圈
                 p=0.7, same_on_batch=False, keepdim=True):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.src_radius, self.angle_range = src_radius, angle_range
        self.alpha_range, self.gamma_range = alpha, gamma
        self.color_range, self.max_flares = color_range, max_flares
        self.register_buffer("_grid", None, persistent=False)

    def generate_parameters(self, batch_shape):
        B, _, H, W = batch_shape
        dev = self.device
        F = self.max_flares

        # 每张图随机本次光圈数量 [1, F]
        n_f = torch.randint(1, F + 1, (B,), device=dev)

        # 位置与形状（沿用你的角度采样方式）
        theta = torch.empty(B, F, device=dev).uniform_(
            self.angle_range[0] * 2 * math.pi, self.angle_range[1] * 2 * math.pi)
        cx = (W * torch.cos(theta)).round()  # (B,F)
        cy = (H * torch.sin(theta)).round()  # (B,F)
        radius = torch.randint(self.src_radius // 2, self.src_radius + 1,
                               (B, F), device=dev)  # (B,F)

        # 强度与形状
        alpha = torch.empty(B, F, device=dev).uniform_(*self.alpha_range)
        gamma = torch.empty(B, F, device=dev).uniform_(*self.gamma_range)

        # 颜色（每圈随机一个 RGB）
        lo, hi = self.color_range
        tint = torch.empty(B, F, 3, device=dev).uniform_(lo, hi)

        return dict(n_f=n_f, cx=cx, cy=cy, radius=radius, alpha=alpha, gamma=gamma, tint=tint)

    def apply_transform(self, x, p, flags=None, transform=None):
        # 设备/精度对齐
        p = _to_same_device(p, x.device)
        B, C, H, W = x.shape
        x_dtype = x.dtype

        # 缓存网格（只在分辨率变更时创建一次）；用时 cast 到 x 的 dtype
        if (self._grid is None) or self._grid.shape[-2:] != (H, W):
            yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                                    torch.arange(W, device=x.device), indexing="ij")
            self._grid = torch.stack((yy, xx), 0)  # float32 buffer
        yy = self._grid[0].to(device=x.device, dtype=x_dtype)
        xx = self._grid[1].to(device=x.device, dtype=x_dtype)

        y = x  # 在输出上逐圈累加
        F = p["cx"].shape[1]
        valid = (torch.arange(F, device=x.device)[None, :] < p["n_f"][:, None])

        for f in range(F):
            if not valid[:, f].any():
                continue
            cx = p["cx"][:, f].to(x_dtype).view(B, 1, 1)
            cy = p["cy"][:, f].to(x_dtype).view(B, 1, 1)
            rad = p["radius"][:, f].to(x_dtype).view(B, 1, 1)
            alp = p["alpha"][:, f].to(x_dtype).view(B, 1, 1)
            gam = p["gamma"][:, f].to(x_dtype).view(B, 1, 1)
            tint = p["tint"][:, f].to(x_dtype).view(B, 3, 1, 1)  # (B,3,1,1)

            # 用 dist^2 近似，避免 sqrt，快且省： mask = relu(1 - dist2/R^2)^gamma * alpha
            dist2 = (xx[None] - cx) ** 2 + (yy[None] - cy) ** 2
            base = 1.0 - dist2 / (rad ** 2 + 1e-6)
            mask = base.clamp_(0, 1).pow_(gam).mul_(alp)  # (B,H,W)

            # 补通道维
            mask = mask.unsqueeze(1)  # -> (B,1,H,W)
            y.addcmul_(mask, tint)

        return y.clamp_(0, 1)


# -----------------------------------------------------------
# 2) Coarse Dropout
# -----------------------------------------------------------
class RandomCoarseDropoutK(AugmentationBase2D):
    def __init__(self, num_holes=(2, 5), hole_hw=(8, 32),
                 p=0.3, same_on_batch=False, keepdim=True):
        super().__init__(p, same_on_batch, keepdim)
        self.n_range, self.hw_range = num_holes, hole_hw
        self.register_buffer("_grid", None, persistent=False)

    def generate_parameters(self, batch_shape):
        B, _, H, W = batch_shape
        dev = self.device
        max_h = self.n_range[1]
        return dict(
            n_h=torch.randint(*self.n_range, (B,), device=dev),
            hs=torch.randint(*self.hw_range, (B, max_h), device=dev),
            ws=torch.randint(*self.hw_range, (B, max_h), device=dev),
            y0=torch.randint(0, H, (B, max_h), device=dev),
            x0=torch.randint(0, W, (B, max_h), device=dev)
        )

    def apply_transform(self, x, p, flags=None, transform=None):
        p = _to_same_device(p, x.device)
        B, _, H, W = x.shape
        if (self._grid is None) or self._grid.shape[-2:] != (H, W):
            rows = torch.arange(H, device=x.device)[:, None]
            cols = torch.arange(W, device=x.device)[None, :]
            self._grid = torch.stack((rows.repeat(1, W), cols.repeat(H, 1)), 0)
        rows, cols = self._grid

        max_h = self.n_range[1]
        y0, x0 = p["y0"][:, :, None, None], p["x0"][:, :, None, None]
        y1 = (y0 + p["hs"][:, :, None, None]).clamp(max=H)
        x1 = (x0 + p["ws"][:, :, None, None]).clamp(max=W)
        valid = torch.arange(max_h, device=x.device)[None, :, None, None] < p["n_h"][:, None, None, None]
        rect = (rows >= y0) & (rows < y1) & (cols >= x0) & (cols < x1) & valid
        mask = rect.any(1, keepdim=True).float()
        return x * (1 - mask)


# -----------------------------------------------------------
# 3) LED Stripe
# -----------------------------------------------------------
# class RandomLEDStripeK(AugmentationBase2D):
#     def __init__(self, max_stripes=10, width_range=(0.001, 0.003),
#                  color_alpha=(0.5, 1.2), p=1.0,  # 先设1.0方便验证
#                  same_on_batch=False, keepdim=True):
#         super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
#         self.max_s, self.wr, self.color_alpha = max_stripes, width_range, color_alpha
#         self.register_buffer("_grid", None, persistent=False)
#
#     def generate_parameters(self, batch_shape):
#         B, _, H, W = batch_shape
#         dev = self.device
#         return dict(
#             n_s=torch.randint(1, self.max_s+1, (B,), device=dev),
#             theta=torch.rand(B, self.max_s, device=dev) * math.pi,
#             width=(self.wr[0] + torch.rand(B, self.max_s, device=dev) *
#                    (self.wr[1]-self.wr[0])) * max(H, W),
#             alpha=(self.color_alpha[0] + torch.rand(B, self.max_s, device=dev) *
#                    (self.color_alpha[1]-self.color_alpha[0])),
#             color=torch.rand(B, self.max_s, 3, device=dev),
#             # 先产生 [-1,1] 的无量纲 offset，具体范围到 apply 再缩放
#             offset_u = torch.rand(B, self.max_s, device=dev) * 2 - 1
#         )
#
#     def apply_transform(self, x, p, flags=None, transform=None):
#         # 确保在 GPU 上 & dtype 合适
#         if x.dtype == torch.uint8:
#             x = x.float() / 255.0
#
#         p = _to_same_device(p, x.device)
#         B, _, H, W = x.shape
#
#         # 1) 居中网格（关键）
#         if (self._grid is None) or self._grid.shape[-2:] != (H, W):
#             yy, xx = torch.meshgrid(
#                 torch.arange(H, device=x.device),
#                 torch.arange(W, device=x.device),
#                 indexing="ij"
#             )
#             yy = yy.float() - (H - 1) / 2.0
#             xx = xx.float() - (W - 1) / 2.0
#             self._grid = torch.stack((yy, xx), 0)
#         yy, xx = self._grid
#
#         # 2) 按角度计算投影，并给每条stripe设“合理的 offset 范围”
#         nx, ny = torch.cos(p["theta"]), torch.sin(p["theta"])          # (B,S)
#         proj = nx[:, :, None, None] * xx + ny[:, :, None, None] * yy   # (B,S,H,W)
#         # 居中坐标下，proj 的范围大致是 [-R, +R]，R≈|nx|*(W-1)/2 + |ny|*(H-1)/2
#         R = (nx.abs() * (W - 1) / 2.0 + ny.abs() * (H - 1) / 2.0)      # (B,S)
#         offset = p["offset_u"] * R                                     # (B,S)
#
#         # 3) 生成条纹强度
#         dist = (proj - offset[:, :, None, None]).abs()
#         stripe = (1 - dist / p["width"][:, :, None, None]).clamp_(0, 1) * p["alpha"][:, :, None, None]
#
#         # 4) 只保留前 n_s 条
#         valid = (torch.arange(self.max_s, device=x.device)[None, :] < p["n_s"][:, None]).float()
#         stripe = stripe * valid[:, :, None, None]
#
#         # 5) 上色并叠加
#         add_rgb = (stripe[..., None] * p["color"][:, :, None, None, :]).sum(1)  # (B,H,W,3)
#         y = (x + add_rgb.permute(0, 3, 1, 2)).clamp_(0, 1)
#         return y
#
# class RandomLEDStripeK(AugmentationBase2D):
#     def __init__(self, max_stripes=10, width_range=(0.02, 0.07),
#                  color_alpha=(0.5, 1.2), p=0.5,
#                  same_on_batch=False, keepdim=True):
#         super().__init__(p, same_on_batch, keepdim)
#         self.max_s, self.wr, self.color_alpha = max_stripes, width_range, color_alpha
#         self.register_buffer("_grid", None, persistent=False)
#
#     def generate_parameters(self, batch_shape):
#         B, _, H, W = batch_shape
#         dev = self.device
#         return dict(
#             n_s=torch.randint(1, self.max_s+1, (B,), device=dev),
#             theta=torch.rand(B, self.max_s, device=dev) * math.pi,
#             width=(self.wr[0] + torch.rand(B, self.max_s, device=dev) *
#                    (self.wr[1]-self.wr[0])) * max(H, W),
#             alpha=(self.color_alpha[0] + torch.rand(B, self.max_s, device=dev) *
#                    (self.color_alpha[1]-self.color_alpha[0])),
#             color=torch.rand(B, self.max_s, 3, device=dev),
#             offset=torch.rand(B, self.max_s, device=dev) * math.hypot(H, W) - math.hypot(H, W)/2
#         )
#
#     def apply_transform(self, x, p, flags=None, transform=None):
#         p = _to_same_device(p, x.device)
#         B, _, H, W = x.shape
#         if (self._grid is None) or self._grid.shape[-2:] != (H, W):
#             yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
#                                     torch.arange(W, device=x.device), indexing="ij")
#             self._grid = torch.stack((yy.float(), xx.float()), 0)
#         yy, xx = self._grid
#
#         nx, ny = torch.cos(p["theta"]), torch.sin(p["theta"])
#         proj = nx[:, :, None, None] * xx + ny[:, :, None, None] * yy
#         dist = (proj - p["offset"][:, :, None, None]).abs()
#         stripe = (1 - dist / p["width"][:, :, None, None]).clamp_(0, 1) * p["alpha"][:, :, None, None]
#
#         valid = torch.arange(self.max_s, device=x.device)[None, :] < p["n_s"][:, None]
#         stripe = stripe * valid[:, :, None, None]
#
#         add_rgb = (stripe[..., None] * p["color"][:, :, None, None, :]).sum(1)
#         return (x + add_rgb.permute(0, 3, 1, 2)).clamp(0, 1)


# -----------------------------------------------------------
# 4) Low‑Light
# -----------------------------------------------------------
class RandomLowLightK(AugmentationBase2D):
    def __init__(self, gamma_range=(0.1, 1.5), mult_range=(0.2, 0.8),
                 noise_std=(0.005, 0.02), p=0.4,
                 same_on_batch=False, keepdim=True):
        super().__init__(p, same_on_batch, keepdim)
        self.gamma_range, self.mult_range, self.noise_std = gamma_range, mult_range, noise_std

    def generate_parameters(self, batch_shape):
        B = batch_shape[0]
        dev = self.device
        return dict(
            gamma=torch.empty(B, device=dev).uniform_(*self.gamma_range),
            mult=torch.empty(B, device=dev).uniform_(*self.mult_range),
            noise=torch.empty(B, device=dev).uniform_(*self.noise_std)
        )

    def apply_transform(self, x, p, flags=None, transform=None):
        p = _to_same_device(p, x.device)
        noise = torch.randn_like(x) * p["noise"][:, None, None, None]
        return (x.pow(p["gamma"][:, None, None, None]) * p["mult"][:, None, None, None] + noise).clamp_(0, 1)


# ---------- C. helper ----------
def make_oneof(*mods):
    return ImageSequential(*mods, random_apply=1, same_on_batch=False)


class FastBCS(AugmentationBase2D):
    def __init__(self, b=(0.1, 2.0), c=(0.6, 2.0), s=(0.6, 2.0),
                 p=1.0, same_on_batch=True, keepdim=True):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.b, self.c, self.s = b, c, s
        self.register_buffer("_w", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1), persistent=False)

    def generate_parameters(self, batch_shape):
        B, _, _, _ = batch_shape
        dev = self.device
        return dict(
            br=torch.empty(B, device=dev).uniform_(*self.b),
            ct=torch.empty(B, device=dev).uniform_(*self.c),
            st=torch.empty(B, device=dev).uniform_(*self.s),
        )

    def apply_transform(self, x, p, flags=None, transform=None):
        if x.dtype == torch.uint8:
            x = x.float().div_(255)
        p = _to_same_device(p, x.device)
        w = self._w.to(x.device)
        B = x.shape[0]
        br = p["br"].view(B, 1, 1, 1)
        ct = p["ct"].view(B, 1, 1, 1)
        st = p["st"].view(B, 1, 1, 1)
        y = (x - 0.5) * ct + 0.5  # Contrast（围绕0.5）
        y = y * br  # Brightness（乘法亮度）
        gray = (y * w).sum(1, keepdim=True)
        y = gray + st * (y - gray)  # Saturation
        return y.clamp_(0, 1)


class RandomGaussianNoiseStdK(AugmentationBase2D):
    """GaussianNoiseStd扰动: p=0.25, std ~U(0.0, 0.15)"""

    def __init__(self, std_range=(0.0, 0.15), p=0.25, same_on_batch=False, keepdim=True):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.std_range = std_range

    def generate_parameters(self, batch_shape):
        B = batch_shape[0]
        dev = self.device
        return dict(std=torch.empty(B, device=dev).uniform_(*self.std_range))

    def apply_transform(self, x, p, flags=None, transform=None):
        if x.dtype == torch.uint8:
            x = x.float().div_(255)
        p = _to_same_device(p, x.device)
        std = p["std"].view(-1, 1, 1, 1)
        noise = torch.randn_like(x) * std
        return (x + noise).clamp_(0, 1)


# ---------- D. Pipeline ----------
class AugPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = ImageSequential(
            make_oneof(
                RandomGaussianNoiseStdK(p=1.0, std_range=(0.0, 0.15)),
                Kaug.RandomMotionBlur(kernel_size=7, angle=(-90., 90.), direction=(-1., 1.), p=1.0),  # 0.046
                FastBCS(p=1.0),
                Kaug.RandomPlanckianJitter(p=1),  # 0.037
                Kaug.RandomChannelDropout(p=1.0),  # 3 0.035
                Kaug.RandomRGBShift()
               # Kaug.RandomRGBShift(),  # 0.039 变色
                # Kaug.RandomGaussianNoise(mean=0., std=0.15, p=1),  # 0.038
                # Kaug.RandomSnow(p=1, snow_coefficient=(0.1, 0.6), brightness=(1.0, 5.0)),  # 2  0.08
                # RandomSunFlareK(p=1)
            )
        )

        # Kaug.Lambda(lambda x: x.clamp_(0,1))
        # self.aug = self.aug.half().to(memory_format=torch.channels_last)
        # self.aug.train()

    @torch.no_grad()
    def forward(self, x): return self.aug(x).clamp_(0, 1)


import cv2, albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


def deg2frac(deg):
    """任意角度（°）→ 0‑1 归一化比例"""
    return (deg % 360) / 360.0


ranges = [(deg2frac(-45), 1.0),  # 315°‑360°（0.875‑1.0）
          (0.0, deg2frac(45))]  # 0°‑45°（0‑0.125）
import numpy as np
import torch.nn.functional as F


def blackout_blocks(rgb, frac=0.20, grid=(12, 12)):
    """把大约 frac 的像素按块置黑。支持 BHWC 或 BCHW；rgb 可为 uint8/float。"""
    is_bhwc = (rgb.dim() == 4 and rgb.size(-1) in (1, 3, 4))
    B = rgb.size(0)
    H, W = (rgb.size(1), rgb.size(2)) if is_bhwc else (rgb.size(-2), rgb.size(-1))
    gh, gw = grid

    # 低分辨率伯努利采样 -> 最近邻上采样成块状 mask
    drop_lo = (torch.rand(B, 1, gh, gw, device=rgb.device) < frac).float()
    drop = F.interpolate(drop_lo, size=(H, W), mode="nearest")  # [B,1,H,W]
    if is_bhwc:
        drop = drop.permute(0, 2, 3, 1)  # [B,H,W,1]

    # 置黑（通道一起置 0）
    return rgb.masked_fill(drop.bool(), 0)

import os, math, tempfile, hashlib
from PIL import Image
class BaseDigitalTwinEnv(BaseEnv):
    """Base Environment class for easily setting up evaluation digital twins for real2sim and sim2real

    This is based on the [SIMPLER](https://simpler-env.github.io/) and currently has the following tricks for
    making accurate simulated environments of real world datasets

    Greenscreening: Add a greenscreened real image to the background to make the images more realistic and closer to the distribution
    of real world data. To use the functionality in your own custom task you can do the following:

    .. code-block:: python

        class MyTask(BaseDigitalTwinEnv):
            def __init__(self, **kwargs):
                self.rgb_overlay_paths = {"camera_name": "path/to/greenscreen/image.png"}
                super().__init__(**kwargs)
            def _load_scene(self, options: dict):
                # load your objects as usual e.g. a cube at self.cube

                # exclude the robot and cube from the greenscreen process
                self.remove_object_from_greenscreen(self.robot)
                self.remove_object_from_greenscreen(self.cube)

    Use `self.remove_object_from_grenscreen(object: Actor | Link | Articulation)` to exclude those objects from the greenscreen process.

    Table Texture Randomization: Randomize table textures from a folder of texture images.
    The texture is randomized during reconfigure (not on each episode reset), so all episodes
    within the same reconfigure cycle will use the same texture. To use this functionality:

    .. code-block:: python

        class MyTask(BaseDigitalTwinEnv):
            def __init__(self, **kwargs):
                # 启用桌面纹理随机化
                self.use_table_texture_randomization = True
                self.table_texture_dir = r"D:\Project\wood_texture"  # 纹理文件夹路径
                super().__init__(**kwargs)

            # 不需要在 _initialize_episode 中调用，纹理会在 _after_reconfigure 时自动随机化
    """

    rgb_overlay_paths: Dict[str, str] = None
    """dict mapping camera name to the file path of the greenscreening image"""
    _rgb_overlay_images: Dict[str, torch.Tensor] = dict()
    """dict mapping camera name to the image torch tensor"""
    rgb_overlay_mode: str = "background"
    """which RGB overlay mode to use during the greenscreen process. The default is 'background' which enables greenscreening like normal. The other option is 'debug' mode which
    will make the opacity of the original render and greenscreen overlay both 50%. The third option is "none" which will not perform any greenscreening."""

    _objects_to_remove_from_greenscreen: List[Union[Actor, Link]] = []
    _objects_to_remove_from_greenscreen_no_table: List[Union[Actor, Link]] = []
    """list of articulations/actors/links that should be removed from the greenscreen process"""
    _segmentation_ids_to_keep: torch.Tensor = None
    """torch tensor of segmentation ids that reference the objects that should not be greenscreened"""
    _segmentation_ids_to_keep_table: torch.Tensor = None
    table_texture_dir: Union[str, None] = None
    """路径到桌面纹理文件夹，如果为None则不启用纹理随机化"""
    use_table_texture_randomization: bool = False
    """是否启用桌面纹理随机化"""
    _table_textures: List[str] = []
    """桌面纹理文件路径列表"""

    background_images_dir: Union[str, None] = None
    """背景图片文件夹路径，如果为None则不启用背景随机化"""
    _background_images: List[Image.Image] = []
    """背景图片列表（PIL Image格式，存储在CPU内存，已缩放到128x128）"""
    _current_background_batch: torch.Tensor = None
    """当前使用的背景图片batch（tensor格式，shape=(B, H, W, 3)，存储在GPU）"""

    def __init__(self, **kwargs):
        # Load the "greenscreen" image, which is used to overlay the background portions of simulation observation

        if self.rgb_overlay_paths is not None:
            print(self._rgb_overlay_images)
            for camera_name, path in self.rgb_overlay_paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"rgb_overlay_path {path} is not found.")
                self._rgb_overlay_images[camera_name] = cv2.cvtColor(
                    cv2.imread(path), cv2.COLOR_BGR2RGB
                )  # (H, W, 3); float32

        else:
            self._rgb_overlay_images = None
        # assert  1== 2

        # 读取桌面纹理文件
        if self.use_table_texture_randomization and self.table_texture_dir is not None:
            self._table_textures = self._read_table_textures()
            if len(self._table_textures) == 0:
                print(f"⚠️ 警告: 在 {self.table_texture_dir} 中未找到纹理文件，桌面纹理随机化将被禁用")
                self.use_table_texture_randomization = False
            else:
                print(f"✓ 已加载 {len(self._table_textures)} 个桌面纹理文件")

        # 加载背景图片（用于背景随机化，预加载到CPU内存，缩放到128x128以节省内存）
        if self.background_images_dir is not None:
            self._load_background_images()
            if len(self._background_images) == 0:
                print(f"⚠️ 警告: 在 {self.background_images_dir} 中未找到背景图片，背景随机化将被禁用")
                self.background_images_dir = None
            else:
                # 计算内存占用（粗略估算）
                num_images = len(self._background_images)
                memory_mb = num_images * 128 * 128 * 3 / (1024 * 1024)  # 每张128x128x3的uint8图片
                print(f"✓ 已加载 {num_images} 个背景图片（CPU内存，128x128，约 {memory_mb:.1f}MB）")

        # self.aug = A.Compose([
        #     # A.MotionBlur(blur_limit=7, p=0.4),
        #     A.OneOf([
        #         A.RandomSunFlare(src_radius=50, num_flare_circles_range=(2, 5),
        #                          angle_range=(0.0, 0.125), p=1.0),
        #         A.PlanckianJitter(temperature_limit=(3000, 7000), p=1.0),
        #     ], p=0.7),
        #     A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
        #     # ② 颜色抖动（加深饱和度/对比度）
        #     A.RandomBrightnessContrast(0.25, 0.4, p=0.7),
        #     A.HueSaturationValue(10, 40, 15, p=0.7),
        #     A.OneOf([
        #         A.RGBShift(r_shift_limit=20, g_shift_limit=15, b_shift_limit=15, p=0.5),
        #         A.ChannelShuffle(p=0.5)
        #     ], p=0.3),
        #
        #     # ③ 噪声 / 模糊 / 压缩
        #     A.OneOf([
        #         A.MotionBlur(blur_limit=3, p=0.3),
        #         A.GaussNoise(var_limit=(1, 5), noise_scale_factor=0.05, p=0.1),
        #         A.ImageCompression(quality_lower=30, quality_upper=70, p=0.1)
        #     ], p=0.5),
        #
        #     # ④ 天气
        #     A.OneOf([
        #         # A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.05, p=0.1),
        #         A.RandomRain(blur_value=2, brightness_coefficient=0.9, p=0.1),
        #         A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=0.1)
        #     ], p=0.3),
        #
        #     # ⑤ 几何
        #     # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
        #     #                    rotate_limit=10, border_mode=0, p=0.6),
        #     # A.HorizontalFlip(p=0.5),
        #
        #     # ⑥ 遮挡
        #     A.CoarseDropout(num_holes_range=(2, 5),
        #                     hole_height_range=(8, 32),
        #                     hole_width_range=(8, 32),
        #                     fill=0, p=0.3),
        #     A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
        #     ToTensorV2()
        # ])

        self.aug = None
        super().__init__(**kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        """
        Load assets for a digital twin scene in
        """

    def remove_object_from_greenscreen(self, object: Union[Articulation, Actor, Link]):
        """remove an actor/articulation/link from the greenscreen process"""
        if isinstance(object, Articulation):
            for link in object.get_links():
                self._objects_to_remove_from_greenscreen.append(link)
        elif isinstance(object, Actor):
            self._objects_to_remove_from_greenscreen.append(object)
        elif isinstance(object, Link):
            self._objects_to_remove_from_greenscreen.append(object)

    def remove_object_from_greenscreen_table(self, object: Union[Articulation, Actor, Link]):
        """remove an actor/articulation/link from the greenscreen process"""
        if isinstance(object, Articulation):
            for link in object.get_links():
                self._objects_to_remove_from_greenscreen_no_table.append(link)
        elif isinstance(object, Actor):
            self._objects_to_remove_from_greenscreen_no_table.append(object)
        elif isinstance(object, Link):
            self._objects_to_remove_from_greenscreen_no_table.append(object)

    def _read_table_textures(self) -> List[str]:
        """
        读取桌面纹理文件夹中的所有 jpg/jpeg 文件

        Returns:
            纹理文件路径列表
        """
        root_dir = Path(self.table_texture_dir).resolve()
        if not root_dir.exists():
            print(f"⚠️ 警告: 纹理文件夹 {root_dir} 不存在")
            return []

        # 递归查找所有 jpg/jpeg 文件（大小写不敏感）
        jpg_paths = [
            str(p.resolve())
            for p in root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        # 按字母顺序排序
        jpg_paths.sort()
        return jpg_paths

    def _load_background_images(self):
        """
        从文件夹加载所有背景图片，缩放到128x128，存储在CPU内存（PIL Image格式）
        注意：2万张图片约占用980MB CPU内存，但reset时访问会很快
        """
        if self.background_images_dir is None:
            return

        root_dir = Path(self.background_images_dir).resolve()
        if not root_dir.exists():
            print(f"⚠️ 警告: 背景图片文件夹 {root_dir} 不存在")
            return

        # 查找所有图片文件
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [
            str(p.resolve())
            for p in root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        if len(image_paths) == 0:
            return

        # 加载所有图片并缩放到128x128（节省内存）
        print(f"正在加载 {len(image_paths)} 张背景图片...")
        self._background_images = []
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((128, 128), Image.Resampling.LANCZOS)
                self._background_images.append(img)
                # 每1000张打印一次进度
                if (i + 1) % 1000 == 0:
                    print(f"  已加载 {i + 1}/{len(image_paths)} 张图片...")
            except Exception as e:
                print(f"⚠️ 警告: 加载图片 {img_path} 失败: {e}")
                continue

        print(f"✓ 背景图片加载完成")

    def _sample_background_batch(self, batch_size: int, target_h: int, target_w: int,
                                 device: torch.device) -> torch.Tensor:
        """
        采样一批背景图片，每个环境一张不同的背景（从预加载的图片中采样）

        Args:
            batch_size: batch大小（环境数量）
            target_h: 目标高度
            target_w: 目标宽度
            device: 目标设备

        Returns:
            背景图片batch，shape=(batch_size, target_h, target_w, 3)，dtype=torch.uint8
        """
        if len(self._background_images) == 0:
            return None

        # 为每个环境随机选择一张背景图片
        selected_indices = [random.randint(0, len(self._background_images) - 1) for _ in range(batch_size)]

        # 从预加载的图片中采样并resize到目标尺寸
        bg_batch = []
        for idx in selected_indices:
            img = self._background_images[idx]
            # 从128x128 resize到目标尺寸（如果不同）
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img_array = np.array(img)  # (H, W, 3), uint8
            bg_batch.append(img_array)

        # 堆叠成batch
        bg_batch = np.stack(bg_batch, axis=0)  # (B, H, W, 3)
        bg_tensor = torch.from_numpy(bg_batch).to(device)  # (B, H, W, 3), uint8

        return bg_tensor



    def _make_rotated_render_texture(self, texture_path: str, theta: float):
        """
        连续角度：theta in [0,2pi)
        旋转在内存完成 -> create_texture_from_array 上传GPU
        """
        base = self._get_rgba_array_cached(texture_path)

        # PIL 旋转（纯内存）
        deg = float(theta) * 180.0 / math.pi
        img = Image.fromarray(base, mode="RGBA")
        img_rot = img.rotate(deg, resample=Image.BILINEAR, expand=False)

        arr_rot = np.ascontiguousarray(np.asarray(img_rot, dtype=np.uint8))

        # 确保传递的数组为正确的RGBA格式，使用字符串参数 format='rgba'
        tex = sapien.render.RenderTexture2D(
            arr_rot,
            format='RGBA',  # 明确指定格式
            mipmap_levels=1,
            filter_mode='linear',
            address_mode='repeat',
            srgb=False  # 如果需要可以设置为 True，通常用于srgb纹理
        )

        return tex
    def _randomize_table_texture(self):
        """
        随机化桌面纹理

        注意：此方法在 _after_reconfigure 时自动调用，不需要在 _initialize_episode 中手动调用。
        这样可以在同一个 reconfigure 周期内保持相同的纹理，只在 reconfigure 时改变纹理。
        """
        if not self.use_table_texture_randomization or len(self._table_textures) == 0:
            return

        # 获取桌子对象（需要子类提供 table_scene 属性）
        if hasattr(self, 'table_scene') and hasattr(self.table_scene, 'table'):
            table = self.table_scene.table
        elif hasattr(self, 'table'):
            table = self.table
        else:
            print("⚠️ 警告: 未找到 table_scene.table 或 table 属性，无法应用纹理随机化")
            return

        # 处理所有环境，为每个环境随机选择纹理
        b = self.num_envs if hasattr(self, 'num_envs') else 1

        # 使用 _batched_main_rng 来确保可重现性（在 reconfigure 时 _batched_episode_rng 可能还未初始化）
        # 在循环外部生成所有随机数，避免重复生成（性能优化）
        import numpy as np
        if hasattr(self, '_batched_main_rng') and self._batched_main_rng is not None:
            # 使用 ManiSkill 的并行模式：为每个环境生成随机纹理索引
            texture_indices = self._batched_main_rng.choice(len(self._table_textures), size=(b,), replace=True)
            if not isinstance(texture_indices, np.ndarray):
                texture_indices = np.array(texture_indices)
            texture_indices = texture_indices.flatten()

            # 生成所有环境的材质参数随机数（在循环外部生成，避免重复）
            # 完全对齐Isaac Sim的参数范围
            r_vals = self._batched_main_rng.uniform(0.3, 0.6, size=(b,))
            g_vals = self._batched_main_rng.uniform(0.2, 0.4, size=(b,))
            b_vals = self._batched_main_rng.uniform(0.1, 0.2, size=(b,))
            specular_vals = self._batched_main_rng.uniform(0., 1., size=(b,))
            roughness_vals = self._batched_main_rng.uniform(0.3, 0.9, size=(b,))

            # 确保是numpy数组
            if not isinstance(r_vals, np.ndarray):
                r_vals = np.array(r_vals).flatten()
            if not isinstance(g_vals, np.ndarray):
                g_vals = np.array(g_vals).flatten()
            if not isinstance(b_vals, np.ndarray):
                b_vals = np.array(b_vals).flatten()
            if not isinstance(specular_vals, np.ndarray):
                specular_vals = np.array(specular_vals).flatten()
            if not isinstance(roughness_vals, np.ndarray):
                roughness_vals = np.array(roughness_vals).flatten()
        else:
            # 如果 _batched_main_rng 不可用，使用 numpy 随机
            texture_indices = np.random.choice(len(self._table_textures), size=(b,), replace=True)
            r_vals = np.random.uniform(0.3, 0.6, size=(b,))
            g_vals = np.random.uniform(0.2, 0.4, size=(b,))
            b_vals = np.random.uniform(0.1, 0.2, size=(b,))
            specular_vals = np.random.uniform(0., 1., size=(b,))
            roughness_vals = np.random.uniform(0.3, 0.9, size=(b,))


        rot_vals = self._batched_main_rng.uniform(0.0, 2.0 * np.pi, size=(b,))


        # 应用纹理到每个环境
        # table._objs 是一个列表，每个元素对应一个并行环境
        for i in range(min(b, len(table._objs))):
            obj = table._objs[i]
            # 从 numpy 数组中获取标量值并转换为 Python int
            if isinstance(texture_indices, np.ndarray):
                texture_idx = int(texture_indices[i])
            else:
                # 如果是列表或其他类型，直接转换
                val = texture_indices[i]
                if isinstance(val, (list, tuple, np.ndarray)):
                    texture_idx = int(val[0]) if len(val) > 0 else 0
                else:
                    texture_idx = int(val)
            texture_path = self._table_textures[texture_idx]


            render_body_component: sapien.render.RenderBodyComponent = (
                obj.find_component_by_type(sapien.render.RenderBodyComponent)
            )
            if render_body_component is not None:
                for render_shape in render_body_component.render_shapes:


                    for part in render_shape.parts:
                        # 1. 纹理文件（Isaac Sim: diffuse_texture）
                        part.material.set_base_color_texture(
                            sapien.render.RenderTexture2D(texture_path)
                        )

                        # 2. 纹理色调（Isaac Sim: diffuse_tint）
                        # 完全对齐Isaac Sim的参数范围：R(0.3,0.6), G(0.2,0.4), B(0.1,0.2)
                        # 注意：r_vals, g_vals, b_vals 已在循环外部生成，这里直接取第i个值
                        r = float(
                            r_vals[i][0] if isinstance(r_vals[i], np.ndarray) and r_vals[i].size > 1 else r_vals[i])
                        g = float(
                            g_vals[i][0] if isinstance(g_vals[i], np.ndarray) and g_vals[i].size > 1 else g_vals[i])
                        b_val = float(
                            b_vals[i][0] if isinstance(b_vals[i], np.ndarray) and b_vals[i].size > 1 else b_vals[i])
                        part.material.set_base_color([r, g, b_val, 1.0])

                        # 3. 镜面反射级别（Isaac Sim: specular_level）
                        # 完全对齐: (0., 1.)
                        # 注意：specular_vals 已在循环外部生成，这里直接取第i个值
                        specular = float(
                            specular_vals[i][0] if isinstance(specular_vals[i], np.ndarray) and specular_vals[
                                i].size > 1 else specular_vals[i])
                        part.material.specular = specular

                        # 4. 反射粗糙度（Isaac Sim: reflection_roughness_constant）
                        # 完全对齐: (0.3, 0.9)
                        # 注意：roughness_vals 已在循环外部生成，这里直接取第i个值
                        roughness = float(
                            roughness_vals[i][0] if isinstance(roughness_vals[i], np.ndarray) and roughness_vals[
                                i].size > 1 else roughness_vals[i])
                        part.material.roughness = roughness

                        # 清除其他纹理（可选）
                        part.material.set_normal_texture(None)
                        part.material.set_emission_texture(None)
                        part.material.set_transmission_texture(None)
                        part.material.set_metallic_texture(None)
                        part.material.set_roughness_texture(None)

    def _after_reconfigure(self, options: dict):
        # print("Here .... ")
        super()._after_reconfigure(options)

        if self.aug is None:
            self.aug = AugPipeline().to(self.device).eval()

        # 在 reconfigure 时随机化桌面纹理（而不是每次 episode 初始化时）
        if self.use_table_texture_randomization:
            self._randomize_table_texture()
        # print("!!!!", self.rgb_overlay_mode)
        if self.rgb_overlay_mode != "none":
            # after reconfiguration in CPU/GPU sim we have initialized all ids of objects in the scene.
            # and can now get the list of segmentation ids to keep
            per_scene_ids = []
            per_scene_ids_table = []
            for object in self._objects_to_remove_from_greenscreen:
                per_scene_ids.append(object.per_scene_id)
            for object in self._objects_to_remove_from_greenscreen_no_table:
                per_scene_ids_table.append(object.per_scene_id)

            self._segmentation_ids_to_keep = torch.unique(
                torch.concatenate(per_scene_ids)
            )
            self._segmentation_ids_to_keep_table = torch.unique(
                torch.concatenate(per_scene_ids_table)
            )
            # print("0000",  self.rgb_overlay_mode )
            # print("1111", self.device)
            # load the overlay images
            for camera_name in self.rgb_overlay_paths.keys():
                sensor = self._sensor_configs[camera_name]
                if isinstance(sensor, CameraConfig):
                    if isinstance(self._rgb_overlay_images[camera_name], torch.Tensor):
                        continue
                    rgb_overlay_img = cv2.resize(
                        self._rgb_overlay_images[camera_name],
                        (sensor.width, sensor.height),
                    )
                    self._rgb_overlay_images[camera_name] = common.to_tensor(
                        rgb_overlay_img, device=self.device
                    )

        self._objects_to_remove_from_greenscreen = []
        self._objects_to_remove_from_greenscreen_no_table = []

    def _green_sceen_rgb(self, rgb, segmentation, overlay_img):
        """returns green screened RGB data given a batch of RGB and segmentation images and one overlay image"""
        actor_seg = segmentation[..., 0]
        mask = torch.ones_like(actor_seg, device=actor_seg.device, dtype=torch.bool)
        if self._segmentation_ids_to_keep is not None and self._segmentation_ids_to_keep.device != actor_seg.device:
            self._segmentation_ids_to_keep = self._segmentation_ids_to_keep.to(
                actor_seg.device
            )
            self._segmentation_ids_to_keep_table = self._segmentation_ids_to_keep_table.to(
                actor_seg.device
            )
        if self.rgb_overlay_mode == "background" and self._segmentation_ids_to_keep is not None:
            # only overlay the background and keep the foregrounds (robot and target objects) rendered in simulation
            mask[
                torch.isin(
                    actor_seg,
                    self._segmentation_ids_to_keep,
                )
            ] = 0
        else:
            print(self.rgb_overlay_mode, self._segmentation_ids_to_keep)
            assert 1 == 2

        mask = mask[..., None]

        # 如果启用背景随机化，替换overlay_img为随机背景batch
        # if overlay_img is None and self.background_images_dir is not None and self._current_background_batch is not None:
        overlay_img = self._current_background_batch
        # 只在设备不匹配时才移动（避免不必要的开销）
        if overlay_img.device != rgb.device:
            overlay_img = overlay_img.to(rgb.device)
            # 如果尺寸不匹配，resize

        # perform overlay on the RGB observation image
        if "debug" not in self.rgb_overlay_mode:

            # print(rgb.shape, mask.shape, overlay_img.shape)
            rgb = rgb * (~mask) + overlay_img * mask
        else:
            assert 1 == 2
            rgb = rgb * 0.5 + overlay_img * 0.5
            rgb = rgb.to(torch.uint8)

        if self.use_background_randomlization:
            #  print(rgb.shape)
            batch = (rgb.permute(0, 3, 1, 2)
                     .to(device=self.device, dtype=torch.float16)
                     .div_(255).contiguous())
            # batch = rgb.permute(0, 3, 1, 2).contiguous().float().div_(255)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                aug_img = self.aug(batch)  # 跑一次就行
            arg_rgb = (aug_img.mul(255)
                       .clamp_(0, 255)
                       .to(torch.uint8)
                       .permute(0, 2, 3, 1))
            # img = rgb[:, :, ::-1].detach().cpu().numpy()
            # aug_img = self.aug(image=img)["image"]  # (C, H, W) tensor
            # rgb = aug_img.permute(1, 2, 0).to(torch.uint8)
            # print(self.use_background_randomlization)

            del aug_img, batch
        else:
            # print(self.use_background_randomlization)
            assert 1 == 2

        if hasattr(self, '_segmentation_ids_to_keep_table') and self._segmentation_ids_to_keep_table is not None:
            table_mask = torch.isin(actor_seg, self._segmentation_ids_to_keep_table)[..., None]
            mask[table_mask] = 1

        return arg_rgb, rgb, ~mask

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True):
        obs = super()._get_obs_sensor_data(apply_texture_transforms)

        # "greenscreen" process
        # assert 3 == 4
        if self.rgb_overlay_mode == "none":
            return obs

        # print(" self.obs_mode_struct.visual.rgb",  self.obs_mode_struct.visual.rgb,  self.obs_mode_struct.visual.segmentation, self.rgb_overlay_paths)
        # assert  1==2
        if (
                self.obs_mode_struct.visual.rgb
                and self.obs_mode_struct.visual.segmentation
                and self.rgb_overlay_paths is not None
        ):
            # get the actor ids of objects to manipulate; note that objects here are not articulated
            for camera_name in self._rgb_overlay_images.keys():
                # obtain overlay mask based on segmentation info
                assert (
                        "segmentation" in obs[camera_name].keys()
                ), "Image overlay requires segment info in the observation!"

                # 获取overlay图片（可能是固定的或随机的背景）
                if self.background_images_dir is not None:
                    # 使用随机背景（在reset时已采样好batch）
                    overlay_img = None  # 会在_green_sceen_rgb中处理
                elif self._rgb_overlay_images is not None and camera_name in self._rgb_overlay_images:
                    # 使用固定的背景
                    if (
                            self._rgb_overlay_images[camera_name].device
                            != obs[camera_name]["rgb"].device
                    ):
                        self._rgb_overlay_images[camera_name] = self._rgb_overlay_images[
                            camera_name
                        ].to(obs[camera_name]["rgb"].device)
                    overlay_img = self._rgb_overlay_images[camera_name]
                else:
                    # 如果没有背景图片，使用黑色背景
                    overlay_img = torch.zeros_like(obs[camera_name]["rgb"])

                green_screened_rgb, org_green_screened_rgb, attention_mask = self._green_sceen_rgb(
                    obs[camera_name]["rgb"],
                    obs[camera_name]["segmentation"],
                    overlay_img,
                )
                obs[camera_name]["rgb"] = green_screened_rgb
                obs[camera_name]["org_rgb"] = org_green_screened_rgb
                obs[camera_name]["mask"] = attention_mask
        return obs

    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        """
        Take a step through the environment with an action. Actions are automatically clipped to the action space.

        If ``action`` is None, the environment will proceed forward in time without sending any actions/control signals to the agent
        """
        action = self._step_action(action)
        self._elapsed_steps += 1
        info = self.get_info()
        obs = self.get_obs(info, unflattened=True)
        reward = self.get_reward(obs=obs, action=action, info=info)
        obs = self._flatten_raw_obs(obs)
        if "success" in info:
            if "fail" in info:
                terminated = torch.logical_or(info["success"], info["fail"])
            else:
                terminated = info["success"].clone()
        else:
            if "fail" in info:
                terminated = info["fail"].clone()
            else:
                terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        self._last_obs = obs

        # add org rgb into info to avoid too much modification
        sensor_data = obs["sensor_data"]

        org_rgb_images = []
        rgb_mask = []
        # print("????",sensor_data.keys())
        for cam_data in sensor_data.values():
            org_rgb_images.append(cam_data["org_rgb"])
            rgb_mask.append(cam_data["mask"])
        # assert len(org_rgb_images) == 2
        for rgb_index, rgb in enumerate(org_rgb_images):
            info["org_rgb_" + str(rgb_index)] = rgb
            info["mask_" + str(rgb_index)] = rgb_mask[rgb_index]
        #  print("info", info.keys())
        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=bool, device=self.device),
            info,
        )

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def reset(self, seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None):
        # 在reset时随机采样背景图片batch（为每个环境采样不同的背景）

        # 清理旧的batch（释放GPU显存）
        if self._current_background_batch is not None:
            del self._current_background_batch
            self._current_background_batch = None

        # 获取batch size和目标尺寸
        batch_size = self.num_envs if hasattr(self, 'num_envs') else 1

        # 获取目标尺寸
        # 注意：直接使用128x128，因为预加载的图片是128x128，实际使用时会在_green_sceen_rgb中根据实际rgb尺寸动态resize
        target_h, target_w = 128, 128

        # 采样新的背景batch（使用self.device，但会在使用时确保设备匹配）
        bg_batch = self._sample_background_batch(
            batch_size, target_h, target_w, self.device
        )

        self._current_background_batch = bg_batch

        obs, info = super().reset(seed, options)

        sensor_data = obs["sensor_data"]

        org_rgb_images = []
        rgb_mask = []
        # print("????",sensor_data.keys())
        for cam_data in sensor_data.values():
            org_rgb_images.append(cam_data["org_rgb"])
            rgb_mask.append(cam_data["mask"])
        # assert len(org_rgb_images) == 2
        for rgb_index, rgb in enumerate(org_rgb_images):
            info["org_rgb_" + str(rgb_index)] = rgb
            info["mask_" + str(rgb_index)] = rgb_mask[rgb_index]

        return obs, info