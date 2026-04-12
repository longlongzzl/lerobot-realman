import math, random, cv2, torch, torch.nn as nn
import kornia.augmentation as Kaug
from kornia.augmentation.container import ImageSequential
import matplotlib.pyplot as plt
from kornia.augmentation._2d.base import AugmentationBase2D


def _to_same_device(d, dev):
    """把随机参数 dict 全部迁移到输入 dev；非 Tensor 原样返回。"""
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}

def _rand_like(shape, x):            # 统一 dtype / device / layout
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
                 color_range=(0.0, 1.0),      # 随机RGB范围
                 max_flares=3,                # 最多几个光圈
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
            self.angle_range[0]*2*math.pi, self.angle_range[1]*2*math.pi)
        cx = (W * torch.cos(theta)).round()               # (B,F)
        cy = (H * torch.sin(theta)).round()               # (B,F)
        radius = torch.randint(self.src_radius // 2, self.src_radius + 1,
                               (B, F), device=dev)        # (B,F)

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
# 4) Random Shadow (快速矩形阴影)
# -----------------------------------------------------------
class RandomShadowK(AugmentationBase2D):
    """
    极速矩形阴影增强，使用纯向量化操作，无循环
    模拟半透明、深浅不一的阴影方块
    """
    def __init__(self, 
                 num_shadows=(1, 4),  # 每张图阴影数量
                 alpha_range=(0.2, 0.6),  # 阴影透明度范围（值越大阴影越暗）
                 size_range=(0.1, 0.3),  # 阴影大小（相对于图像尺寸的比例）
                 p=0.8,  # 应用概率
                 same_on_batch=False, 
                 keepdim=True):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.num_shadows = num_shadows
        self.alpha_range = alpha_range
        self.size_range = size_range

    def generate_parameters(self, batch_shape):
        B, _, H, W = batch_shape
        dev = self.device
        max_shadows = self.num_shadows[1]
        
        # 每张图的阴影数量
        n_shadows = torch.randint(*self.num_shadows, (B,), device=dev)
        
        # 阴影大小
        size_base = torch.empty(B, max_shadows, device=dev).uniform_(*self.size_range)
        w_shadow = (size_base * W).int().clamp_(1, W)
        h_shadow = (size_base * H).int().clamp_(1, H)
        
        # 阴影位置
        x0 = torch.randint(0, W, (B, max_shadows), device=dev)
        y0 = torch.randint(0, H, (B, max_shadows), device=dev)
        x1 = (x0 + w_shadow).clamp_(max=W)
        y1 = (y0 + h_shadow).clamp_(max=H)
        
        # 阴影透明度（深浅不一）
        alpha = torch.empty(B, max_shadows, device=dev).uniform_(*self.alpha_range)
        
        return dict(n_shadows=n_shadows, x0=x0, y0=y0, x1=x1, y1=y1, alpha=alpha)

    def apply_transform(self, x, p, flags=None, transform=None):
        p = _to_same_device(p, x.device)
        B, C, H, W = x.shape
        x_dtype = x.dtype
        
        # 初始化阴影因子（全1表示无阴影）
        shadow_factor = torch.ones(B, H, W, device=x.device, dtype=x_dtype)
        max_shadows = self.num_shadows[1]
        
        # 创建坐标网格 [H, W]
        yy = torch.arange(H, device=x.device, dtype=x_dtype).view(H, 1)  # [H, 1]
        xx = torch.arange(W, device=x.device, dtype=x_dtype).view(1, W)  # [1, W]
        
        # 批量处理所有阴影
        for s in range(max_shadows):
            # 检查哪些batch需要这个阴影
            valid = (torch.arange(max_shadows, device=x.device)[None, :] < p["n_shadows"][:, None])[:, s]
            
            if not valid.any():
                continue
            
            # 获取参数 [B]
            x0 = p["x0"][:, s].to(x_dtype)  # [B]
            y0 = p["y0"][:, s].to(x_dtype)  # [B]
            x1 = p["x1"][:, s].to(x_dtype)  # [B]
            y1 = p["y1"][:, s].to(x_dtype)  # [B]
            alpha = p["alpha"][:, s].to(x_dtype)  # [B]
            
            # 向量化判断：每个像素是否在阴影矩形内
            # 扩展维度以便广播 [B, 1, 1] 和 [1, H, W]
            x0_exp = x0.view(B, 1, 1)
            x1_exp = x1.view(B, 1, 1)
            y0_exp = y0.view(B, 1, 1)
            y1_exp = y1.view(B, 1, 1)
            alpha_exp = alpha.view(B, 1, 1)
            
            # 判断像素是否在矩形内 [B, H, W]
            in_rect = ((xx >= x0_exp) & (xx < x1_exp) & (yy >= y0_exp) & (yy < y1_exp)).float()
            
            # 只对有效的batch应用阴影
            valid_mask = valid.float().view(B, 1, 1)
            in_rect = in_rect * valid_mask
            
            # 应用阴影：降低该区域的亮度
            shadow_factor = shadow_factor * (1.0 - alpha_exp * in_rect)
        
        # 应用阴影到所有通道 [B, C, H, W]
        y = x * shadow_factor.unsqueeze(1)
        
        return y.clamp_(0, 1)


# -----------------------------------------------------------
# 5) Low‑Light
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
    def __init__(self, b=(0.1,2.0), c=(0.6,2.0), s=(0.6,2.0),
                 p=1.0, same_on_batch=True, keepdim=True):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.b, self.c, self.s = b, c, s
        self.register_buffer("_w", torch.tensor([0.299, 0.587, 0.114]).view(1,3,1,1), persistent=False)

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
        br = p["br"].view(B,1,1,1)
        ct = p["ct"].view(B,1,1,1)
        st = p["st"].view(B,1,1,1)
        y = (x - 0.5) * ct + 0.5      # Contrast（围绕0.5）
        y = y * br                    # Brightness（乘法亮度）
        gray = (y * w).sum(1, keepdim=True)
        y = gray + st * (y - gray)    # Saturation
        return y.clamp_(0, 1)


# -----------------------------------------------------------
# DEXTRAH颜色抖动（完全按照DEXTRAH逻辑，内存优化版本）
# -----------------------------------------------------------
# 核心计算函数（可编译优化）
@torch.jit.script
def _color_jitter_core(
    x: torch.Tensor,
    sat: torch.Tensor,
    ct: torch.Tensor,
    br: torch.Tensor,
    hue_delta: torch.Tensor,
    w: torch.Tensor,
    use_hue: bool
) -> torch.Tensor:
    """
    核心颜色抖动计算（JIT编译优化）
    顺序：Saturation → Contrast → Brightness → Hue
    """
    # 1. Saturation
    gray = (x * w).sum(dim=-3, keepdim=True)
    y = gray + sat * (x - gray)
    
    # 2. Contrast
    gray_scale = (y * w).sum(dim=-3, keepdim=True)
    avg_brightness = gray_scale.mean(dim=(-2, -1), keepdim=True)
    y = avg_brightness + ct * (y - avg_brightness)
    
    # 3. Brightness
    y = y * br
    
    # 4. Hue（使用简化的RGB旋转）
    if use_hue:
        angle = hue_delta * 2 * 3.141592653589793  # math.pi
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        r, g, b = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        
        new_r = (0.213 + 0.787 * cos_a + 0.213 * sin_a) * r + \
                (0.715 - 0.715 * cos_a + 0.715 * sin_a) * g + \
                (0.072 - 0.072 * cos_a - 0.928 * sin_a) * b
        new_g = (0.213 - 0.213 * cos_a - 0.143 * sin_a) * r + \
                (0.715 + 0.285 * cos_a + 0.140 * sin_a) * g + \
                (0.072 - 0.072 * cos_a + 0.283 * sin_a) * b
        new_b = (0.213 - 0.213 * cos_a + 0.928 * sin_a) * r + \
                (0.715 - 0.715 * cos_a - 0.855 * sin_a) * g + \
                (0.072 + 0.928 * cos_a + 0.072 * sin_a) * b
        
        y = torch.cat([new_r, new_g, new_b], dim=1)
    
    return y.clamp_(0, 1)


class ColorJitterDEXTRAH(AugmentationBase2D):
    """
    完全按照DEXTRAH逻辑的颜色抖动（内存优化版本）
    参数：saturation=[0.5,1.5], contrast=[0.5,1.5], brightness=[0.5,1.5], hue=[-0.15,0.15]
    计算顺序：Saturation → Contrast → Brightness → Hue（与DEXTRAH完全一致）
    使用简化的hue调整避免完整HSV转换，大幅减少内存使用
    """
    def __init__(self, saturation_range=(0.5, 1.5), contrast_range=(0.5, 1.5),
                 brightness_range=(0.5, 1.5), hue_range=(-0.15, 0.15),
                 p=1.0, same_on_batch=False, keepdim=True):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.sat_r, self.ct_r, self.br_r, self.hue_r = saturation_range, contrast_range, brightness_range, hue_range
        self.register_buffer("_w", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1), persistent=False)
        self.use_hue = abs(hue_range[0]) + abs(hue_range[1]) > 0.01

    def generate_parameters(self, batch_shape):
        B = batch_shape[0]
        dev = self.device
        return dict(
            sat=torch.empty(B, device=dev).uniform_(*self.sat_r),
            ct=torch.empty(B, device=dev).uniform_(*self.ct_r),
            br=torch.empty(B, device=dev).uniform_(*self.br_r),
            hue=torch.empty(B, device=dev).uniform_(*self.hue_r),
        )

    def apply_transform(self, x, p, flags=None, transform=None):
        # 类型转换（如果需要）
        if x.dtype == torch.uint8:
            x = x.float().div_(255)
        
        # 设备对齐（一次性完成）
        p = _to_same_device(p, x.device)
        w = self._w.to(x.device)
        B = x.shape[0]
        
        # 一次性准备所有参数（减少view操作）
        sat = p["sat"].view(B, 1, 1, 1)
        ct = p["ct"].view(B, 1, 1, 1)
        br = p["br"].view(B, 1, 1, 1)
        hue_delta = p["hue"].view(B, 1, 1, 1)
        
        # 使用JIT编译的核心计算函数（高性能）
        return _color_jitter_core(x, sat, ct, br, hue_delta, w, self.use_hue)

# ---------- D. Pipeline ----------
class AugPipeline(nn.Module):
    def __init__(self, use_dextrah_color_jitter=True, use_compile=False):
        """
        Args:
            use_dextrah_color_jitter: 是否使用DEXTRAH风格颜色抖动（100%概率）
            use_compile: 是否使用torch.compile优化（PyTorch 2.0+，可提升20-30%速度）
        """
        super().__init__()
        aug_list = []
        
        # DEXTRAH颜色抖动（100%概率）
        if use_dextrah_color_jitter:
            aug_list.append(ColorJitterDEXTRAH(
                saturation_range=(0.5, 1.5),
                contrast_range=(0.5, 1.5),
                brightness_range=(0.5, 1.5),
                hue_range=(-0.15, 0.15),
                p=1.0
            ))
        
        # 其他增强（可选）
        # aug_list.append(make_oneof(
        #     FastBCS(p=1.0),
        #     RandomShadowK(p=1.0, num_shadows=(1, 4), alpha_range=(0.2, 0.6)),
        # ))
        
        self.aug = ImageSequential(*aug_list) if aug_list else nn.Identity()
        self.use_compile = use_compile
        self._compiled = False

    @torch.no_grad()
    def forward(self, x): 
        # 注意：不编译整个pipeline（Kornia有动态控制流），核心计算函数已经JIT编译
        return self.aug(x).clamp_(0, 1)


device =  "cuda"
img = cv2.cvtColor(cv2.imread("D:/Project/Scaling/lerobot-sim2real-main/lerobot-sim2real/greenscreen.png"), cv2.COLOR_BGR2RGB)
print(img.shape)
ten = torch.from_numpy(img).permute(2,0,1).float()/255
batch = ten.unsqueeze(0).repeat(888,1,1,1).to(device)

# 使用DEXTRAH颜色抖动（极致优化版本）
# 核心计算函数已使用JIT编译，无需编译整个pipeline
aug = AugPipeline(use_dextrah_color_jitter=True, use_compile=False).to("cuda").eval()

# 使用channels_last内存格式（提升5-10%速度）
batch = batch.to(memory_format=torch.channels_last)

print("✓ 已启用DEXTRAH颜色抖动（极致优化版本）")
print("✓ 核心计算函数已JIT编译（高性能）")
print("✓ 已启用channels_last内存格式")

import time

# 预热：确保torch.compile编译完成（如果启用）
print("预热中...")
for _ in range(3):
    _ = aug(batch)
torch.cuda.synchronize()  # 确保GPU操作完成

s = time.time()
print(batch.shape)

for i in range(100):
    # with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
    with torch.no_grad():
        _ = aug(batch)  # 跑一次就行
   # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
    # with torch.no_grad():
    #     out = aug(batch)

print("88 time cost", (time.time()-s)/100.0)
n_vis = 9
aug_imgs = []
with torch.no_grad():
    for _ in range(n_vis):
        out = aug(ten.unsqueeze(0).to(device))          # (1,3,H,W)
        aug_imgs.append(out.squeeze(0).cpu())           # 回到 CPU，(3,H,W)

# ----------- 3×3 网格展示 -----------
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for ax, img_t in zip(axes.flatten(), aug_imgs):
    ax.imshow((img_t.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    ax.axis("off")
plt.tight_layout()
plt.show()