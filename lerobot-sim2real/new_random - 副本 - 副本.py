import math, random, cv2, torch, torch.nn as nn
import kornia.augmentation as Kaug
from kornia.augmentation.container import ImageSequential
import matplotlib.pyplot as plt
from kornia.augmentation._2d.base import AugmentationBase2D
import torch.nn.functional as F


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
# 4) Random Shadow (快速多形状阴影)
# -----------------------------------------------------------
class RandomShadowK(AugmentationBase2D):
    """
    极速多形状阴影增强，使用纯向量化操作
    支持：矩形、椭圆、拉长阴影（模拟投射阴影）
    模拟半透明、深浅不一的阴影，更接近真实机械臂阴影
    """

    def __init__(self,
                 num_shadows=(1, 4),  # 每张图阴影数量
                 alpha_range=(0.2, 0.6),  # 阴影透明度范围（值越大阴影越暗）
                 size_range=(0.05, 0.5),  # 阴影大小（相对于图像尺寸的比例）
                 shadow_types=['rectangle', 'ellipse', 'elongated'],  # 阴影类型
                 p=0.8,  # 应用概率
                 same_on_batch=False,
                 keepdim=True):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.num_shadows = num_shadows
        self.alpha_range = alpha_range
        self.size_range = size_range
        self.shadow_types = shadow_types

    def generate_parameters(self, batch_shape):
        B, _, H, W = batch_shape
        dev = self.device
        max_shadows = self.num_shadows[1]

        # 每张图的阴影数量
        n_shadows = torch.randint(*self.num_shadows, (B,), device=dev)

        # 阴影大小（直接使用整数，避免后续转换）
        # 让宽高有不同的随机值，避免都是正方形
        # 为了模拟机械臂投射阴影，让阴影更倾向于拉长形状
        size_base_w = torch.empty(B, max_shadows, device=dev).uniform_(*self.size_range)
        size_base_h = torch.empty(B, max_shadows, device=dev).uniform_(*self.size_range)

        # 进一步增加宽高差异：让其中一个维度更小，模拟拉长阴影
        # 随机决定是横向拉长还是纵向拉长
        is_horizontal = torch.rand(B, max_shadows, device=dev) > 0.5
        # 如果是横向拉长，宽度更大，高度更小；反之亦然
        w_mult = torch.where(is_horizontal,
                             torch.empty(B, max_shadows, device=dev).uniform_(1.2, 2.5),  # 横向拉长：宽更大
                             torch.empty(B, max_shadows, device=dev).uniform_(0.4, 0.8))  # 纵向拉长：宽更小
        h_mult = torch.where(is_horizontal,
                             torch.empty(B, max_shadows, device=dev).uniform_(0.4, 0.8),  # 横向拉长：高更小
                             torch.empty(B, max_shadows, device=dev).uniform_(1.2, 2.5))  # 纵向拉长：高更大

        w_shadow = (size_base_w * W * w_mult).int().clamp_(1, W)
        h_shadow = (size_base_h * H * h_mult).int().clamp_(1, H)

        # 阴影位置（整数坐标，直接生成左上角）
        x0 = torch.randint(0, W, (B, max_shadows), device=dev)
        y0 = torch.randint(0, H, (B, max_shadows), device=dev)
        x1 = (x0 + w_shadow).clamp_(max=W)
        y1 = (y0 + h_shadow).clamp_(max=H)

        # 阴影透明度（深浅不一）
        alpha = torch.empty(B, max_shadows, device=dev).uniform_(*self.alpha_range)

        # 旋转角度（0-360度，转换为弧度）
        angle = torch.empty(B, max_shadows, device=dev).uniform_(0, 2 * math.pi)

        return dict(n_shadows=n_shadows, x0=x0, y0=y0, x1=x1, y1=y1, alpha=alpha, angle=angle)

    def apply_transform(self, x, p, flags=None, transform=None):
        p = _to_same_device(p, x.device)
        B, C, H, W = x.shape
        x_dtype = x.dtype

        # 初始化阴影因子（全1表示无阴影）
        shadow_factor = torch.ones(B, H, W, device=x.device, dtype=x_dtype)
        max_shadows = self.num_shadows[1]

        # 创建坐标网格 [H, W] - 缓存网格避免重复创建
        if not hasattr(self, '_grid_cache') or self._grid_cache is None:
            self._grid_cache = {}
        cache_key = (H, W, x.device)
        if cache_key not in self._grid_cache:
            yy = torch.arange(H, device=x.device, dtype=x_dtype).view(H, 1)  # [H, 1]
            xx = torch.arange(W, device=x.device, dtype=x_dtype).view(1, W)  # [1, W]
            self._grid_cache[cache_key] = (xx, yy)
        xx, yy = self._grid_cache[cache_key]

        # 预计算所有阴影的有效性 [B, max_shadows]
        valid_all = torch.arange(max_shadows, device=x.device)[None, :] < p["n_shadows"][:, None]  # [B, max_shadows]

        # 批量处理所有阴影 - 简化版本，保持速度
        for s in range(max_shadows):
            valid = valid_all[:, s]  # [B]

            if not valid.any():
                continue

            # 获取参数 [B] - 批量转换类型
            x0 = p["x0"][:, s].to(x_dtype)  # [B]
            y0 = p["y0"][:, s].to(x_dtype)  # [B]
            x1 = p["x1"][:, s].to(x_dtype)  # [B]
            y1 = p["y1"][:, s].to(x_dtype)  # [B]
            alpha = p["alpha"][:, s].to(x_dtype)  # [B]
            angle = p["angle"][:, s].to(x_dtype)  # [B] 已经是0-2π（0-360度）

            # 计算矩形中心和半宽高（使用乘法代替除法）
            cx = (x0 + x1) * 0.5  # [B]
            cy = (y0 + y1) * 0.5  # [B]
            w = (x1 - x0) * 0.5  # [B] 半宽
            h = (y1 - y0) * 0.5  # [B] 半高

            # 预计算cos/sin（只计算一次）
            cos_a = torch.cos(angle)  # [B]
            sin_a = torch.sin(angle)  # [B]

            # 扩展维度以便广播
            cx_exp = cx.view(B, 1, 1)  # [B, 1, 1]
            cy_exp = cy.view(B, 1, 1)  # [B, 1, 1]
            w_exp = w.view(B, 1, 1)  # [B, 1, 1]
            h_exp = h.view(B, 1, 1)  # [B, 1, 1]
            cos_a_exp = cos_a.view(B, 1, 1)  # [B, 1, 1]
            sin_a_exp = sin_a.view(B, 1, 1)  # [B, 1, 1]
            alpha_exp = alpha.view(B, 1, 1)  # [B, 1, 1]

            # 计算相对坐标
            dx = xx - cx_exp  # [B, H, W]
            dy = yy - cy_exp  # [B, H, W]

            # 旋转坐标（向量化操作）
            dx_rot = dx * cos_a_exp + dy * sin_a_exp  # [B, H, W]
            dy_rot = -dx * sin_a_exp + dy * cos_a_exp  # [B, H, W]

            # 判断像素是否在旋转后的矩形内
            in_rect = ((dx_rot.abs() <= w_exp) & (dy_rot.abs() <= h_exp)).to(x_dtype)

            # 只对有效的batch应用阴影
            valid_mask = valid.float().view(B, 1, 1)
            in_rect = in_rect * valid_mask

            # 应用阴影：降低该区域的亮度
            shadow_factor = shadow_factor * (1.0 - alpha_exp * in_rect)

        # 应用阴影到所有通道 [B, C, H, W]
        y = x * shadow_factor.unsqueeze(1)

        return y.clamp_(0, 1)


# -----------------------------------------------------------
# 5) Shadow Blocks (块状阴影)
# -----------------------------------------------------------
def shadow_blocks(rgb, frac=0.20, grid=(12, 12), alpha_range=(0.1, 0.9)):
    """
    把大约 frac 的像素按块叠加阴影效果（而不是完全置黑）
    支持 BHWC 或 BCHW；rgb 可为 uint8/float
    每个块的阴影透明度在 alpha_range 范围内随机

    Args:
        rgb: 输入图像，形状为 [B, H, W, C] 或 [B, C, H, W]
        frac: 被阴影覆盖的块的比例（大约）
        grid: 网格大小 (gh, gw)，决定块的数量
        alpha_range: 阴影透明度范围 (min_alpha, max_alpha)，值越大阴影越暗

    Returns:
        添加了阴影效果的图像
    """
    is_bhwc = (rgb.dim() == 4 and rgb.size(-1) in (1, 3, 4))
    B = rgb.size(0)
    H, W = (rgb.size(1), rgb.size(2)) if is_bhwc else (rgb.size(-2), rgb.size(-1))
    gh, gw = grid

    # 确保rgb是浮点数格式（0-1范围）
    if rgb.dtype == torch.uint8:
        rgb_float = rgb.float() / 255.0
    else:
        rgb_float = rgb.clone()

    # 低分辨率伯努利采样 -> 最近邻上采样成块状 mask
    drop_lo = (torch.rand(B, 1, gh, gw, device=rgb.device) < frac).float()  # [B, 1, gh, gw]

    # 为每个块生成随机的alpha值（阴影透明度）
    alpha_lo = torch.empty(B, 1, gh, gw, device=rgb.device).uniform_(*alpha_range)  # [B, 1, gh, gw]
    # 只对被选中的块应用alpha，其他块alpha=0（无阴影）
    alpha_lo = alpha_lo * drop_lo

    # 上采样到原始分辨率
    alpha = F.interpolate(alpha_lo, size=(H, W), mode="nearest")  # [B, 1, H, W]

    if is_bhwc:
        alpha = alpha.permute(0, 2, 3, 1)  # [B, H, W, 1]

    # 应用阴影：降低阴影区域的亮度
    # shadow_factor = 1 - alpha，alpha越大，阴影越暗
    shadow_factor = 1.0 - alpha

    # 应用阴影到所有通道
    rgb_shadowed = rgb_float * shadow_factor

    # 转换回原始格式
    if rgb.dtype == torch.uint8:
        rgb_shadowed = (rgb_shadowed * 255.0).clamp_(0, 255).to(torch.uint8)

    return rgb_shadowed


# -----------------------------------------------------------
# 6) Low‑Light
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


# ---------- D. Pipeline ----------
class AugPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = ImageSequential(
            # RandomLowLightK(p=0.5),  # 全局夜景先随机应用
            # RandomLEDStripeK(p=0.3),
            make_oneof(
              # Kaug.RandomChannelShuffle(p=1),  # 0.097 色温，有点对比度的感觉
               Kaug.RandomMotionBlur(kernel_size=7,     angle=(-90., 90.),   direction=(-1., 1.), p=1.0),  # 0.046
                 FastBCS(p=1.0),
                 Kaug.RandomPlanckianJitter(p=1),  # 0.037
                  Kaug.RandomChannelDropout(p=1.0),  # 3 0.035
                Kaug.RandomRGBShift(),  # 0.039 变色
                  Kaug.RandomGaussianNoise(mean=0., std=0.05, p=1), # 0.038
            #  Kaug.RandomSnow(p=1, snow_coefficient=(0.1, 0.6), brightness=(1.0, 5.0)),  # 2  0.08
                #       RandomSunFlareK(p=1),
                #RandomShadowK(p=1.0, num_shadows=(1, 4), alpha_range=(0.1, 0.9)),  # 快速矩形阴影
                #               Kaug.RandomRain(number_of_drops=(10, 50), drop_height=(2, 20), drop_width=(-2, 2),  p=1), # 1  0.63
                #                RandomCoarseDropoutK(num_holes=(2, 6), p=1.0), # 0.031

                #   RandomLEDStripeK(color_alpha=(2.0, 4.0),width_range=(0.05, 0.15), p=1.0), # 0.033 # no change !!!!!
            )
        )

        # Kaug.Lambda(lambda x: x.clamp_(0,1))
        # self.aug = self.aug.half().to(memory_format=torch.channels_last)
        # self.aug.train()

    @torch.no_grad()
    def forward(self, x): return self.aug(x).clamp_(0, 1)


device = "cuda"
img = cv2.cvtColor(cv2.imread("D:/Project/Scaling/lerobot-sim2real-main/lerobot-sim2real/greenscreen.png"),
                   cv2.COLOR_BGR2RGB)
print(img.shape)
ten = torch.from_numpy(img).permute(2, 0, 1).float() / 255
batch = ten.unsqueeze(0).repeat(888, 1, 1, 1).to(device)
aug = AugPipeline().to("cuda").eval()

import time

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

# print("88 time cost", (time.time() - s) / 100.0)
# n_vis = 9
# aug_imgs = []
# with torch.no_grad():
#     for _ in range(n_vis):
#        # out = shadow_blocks(ten.unsqueeze(0).to(device), frac=0.20, grid=(12, 12), alpha_range=(0.1, 0.9))
#         #  out = aug(ten.unsqueeze(0).to(device))          # (1,3,H,W)
#        with torch.no_grad():
#            out = aug(batch)
#         aug_imgs.append(out.squeeze(0).cpu())  # 回到 CPU，(3,H,W)
#
# # ----------- 3×3 网格展示 -----------
# fig, axes = plt.subplots(3, 3, figsize=(9, 9))
# for ax, img_t in zip(axes.flatten(), aug_imgs):
#     ax.imshow((img_t.permute(1, 2, 0).numpy() * 255).astype("uint8"))
#     ax.axis("off")
# plt.tight_layout()
# plt.show()