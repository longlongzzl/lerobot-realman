import math, torch, time
import torchvision.transforms.v2 as v2
from torch import nn

# ─────────────────────────────────────────────────────────────
# A. 自定义向量化增广（兼容 v2）
# ─────────────────────────────────────────────────────────────
class RandomSunFlare(nn.Module):
    r"""在图像边缘随机叠加软边缘日晕 / 镜头光斑。"""
    def __init__(self, src_radius=50, angle_range=(0.0, 0.125), p=0.7):
        super().__init__()
        self.src_radius, self.angle_range, self.p = src_radius, angle_range, p

    def forward(self, x: torch.Tensor):
        # x : (B,3,H,W) float32 0‑1
        B, _, H, W, dev = *x.shape, x.device
        apply = torch.rand(B, device=dev) < self.p
        if not apply.any():
            return x

        theta  = torch.empty(B, device=dev).uniform_(*self.angle_range) * 2*math.pi
        radius = torch.randint(self.src_radius//2, self.src_radius+1, (B,), device=dev)
        alpha  = torch.rand(B, device=dev).uniform_(0.4, 1.0)
        gamma  = torch.rand(B, device=dev).uniform_(1.5, 3.0)

        cx = (W * torch.cos(theta)).round().long()
        cy = (H * torch.sin(theta)).round().long()

        yy, xx = torch.meshgrid(
            torch.arange(H, device=dev),
            torch.arange(W, device=dev), indexing="ij"
        )                                                   # (H,W)

        dist = torch.sqrt((xx[None]-cx[:,None,None]).pow(2) +
                          (yy[None]-cy[:,None,None]).pow(2))          # (B,H,W)
        mask = (1 - dist / radius[:,None,None]).clamp_(0, 1)
        mask = mask.pow(gamma[:,None,None]) * alpha[:,None,None]      # (B,H,W)
        mask = mask * apply[:,None,None]

        return (x + mask[:,None]).clamp_(0, 1)


class RandomCoarseDropout(nn.Module):
    r"""随机挖洞遮挡 (Cutout)，无 for‑loop，支持批量不同参数。"""
    def __init__(self, num_holes=(2, 5), hole_hw=(8, 32), p=0.3):
        super().__init__()
        self.n_range, self.hw_range, self.p = num_holes, hole_hw, p

    def forward(self, x):
        B, _, H, W, dev = *x.shape, x.device
        apply = torch.rand(B, device=dev) < self.p
        if not apply.any():
            return x

        max_holes = self.n_range[1]
        n_holes   = torch.randint(*self.n_range, (B,), device=dev)

        hs, ws = [torch.randint(*self.hw_range, (B, max_holes), device=dev) for _ in range(2)]
        y0 = torch.randint(0, H, (B, max_holes), device=dev)
        x0 = torch.randint(0, W, (B, max_holes), device=dev)
        y1 = (y0 + hs).clamp(max=H)
        x1 = (x0 + ws).clamp(max=W)

        rows = torch.arange(H, device=dev)[None, None, :, None]
        cols = torch.arange(W, device=dev)[None, None, None, :]
        valid = (torch.arange(max_holes, device=dev)[None] < n_holes[:, None])[:, :, None, None]

        rect = ((rows >= y0[:, :, None, None]) & (rows < y1[:, :, None, None]) &
                (cols >= x0[:, :, None, None]) & (cols < x1[:, :, None, None])) & valid
        mask = rect.any(1).float() * apply[:, None, None]             # (B,H,W)
        return x * (1 - mask[:, None])


class RandomLEDStripe(nn.Module):
    r"""在夜店/隧道场景常见的彩色 LED 条带光。"""
    def __init__(self, max_stripes=10, width_range=(0.02, 0.07),
                 p=0.5, color_alpha=(0.5, 1.2)):
        super().__init__()
        self.max_s, self.wr, self.p, self.ca = max_stripes, width_range, p, color_alpha

    def forward(self, x):
        B, _, H, W, dev = *x.shape, x.device
        apply = torch.rand(B, device=dev) < self.p
        if not apply.any():
            return x

        max_s = self.max_s
        n_s   = torch.randint(1, max_s+1, (B,), device=dev)

        theta  = torch.rand(B, max_s, device=dev) * math.pi
        width  = (self.wr[0] + torch.rand(B, max_s, device=dev) *
                  (self.wr[1] - self.wr[0])) * max(H, W)
        alpha  = (self.ca[0] + torch.rand(B, max_s, device=dev) *
                  (self.ca[1] - self.ca[0]))
        color  = torch.rand(B, max_s, 3, device=dev)

        valid  = (torch.arange(max_s, device=dev)[None] < n_s[:, None])
        alpha  = alpha * valid                                            # 失效条带 α=0

        yy, xx = torch.meshgrid(torch.arange(H, device=dev),
                                torch.arange(W, device=dev), indexing="ij")
        xx, yy = xx.float(), yy.float()
        nx, ny = torch.cos(theta), torch.sin(theta)
        proj   = nx[:, :, None, None] * xx + ny[:, :, None, None] * yy
        offset = torch.rand(B, max_s, device=dev) * math.hypot(H, W) - math.hypot(H, W)/2
        dist   = (proj - offset[:, :, None, None]).abs()
        stripe = (1 - dist / width[:, :, None, None]).clamp_(0, 1) * alpha[:, :, None, None]

        add_rgb = (stripe[..., None] * color[:, :, None, None, :]).sum(1)  # (B,H,W,3)
        return (x + add_rgb.permute(0, 3, 1, 2)).clamp_(0, 1)


class RandomLowLight(nn.Module):
    r"""整幅夜景/低光: γ 校正 + 亮度缩放 + 随机噪声。"""
    def __init__(self, gamma_range=(0.1, 1.5), mult_range=(0.2, 0.8),
                 noise_std=(0.005, 0.02), p=0.4):
        super().__init__()
        self.gmin, self.gmax = gamma_range
        self.mmin, self.mmax = mult_range
        self.nmin, self.nmax = noise_std
        self.p = p

    def forward(self, x):
        if torch.rand(1, device=x.device) > self.p:
            return x
        B, _, _, _ = x.shape
        gamma = torch.empty(B, device=x.device).uniform_(self.gmin, self.gmax)
        mult  = torch.empty(B, device=x.device).uniform_(self.mmin, self.mmax)
        noise = torch.randn_like(x) * torch.empty(
            B, 1, 1, 1, device=x.device).uniform_(self.nmin, self.nmax)

        x = x.pow(gamma[:, None, None, None])
        x = x * mult[:, None, None, None] + noise
        return x.clamp_(0, 1)

# ─────────────────────────────────────────────────────────────
# B. 帮助函数：随机抽 K 个增广
# ─────────────────────────────────────────────────────────────
class RandomSubset(nn.Module):
    r"""从给定列表中随机抽 k 个依次执行。等价于 kornia 的 ImageSequential(..., random_apply=k)."""
    def __init__(self, transforms, k):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.k = k

    def forward(self, x):
        idx = torch.randperm(len(self.transforms), device=x.device)[:self.k]
        for i in idx:
            x = self.transforms[i](x)
        return x

def make_oneof(k, *mods):
    return RandomSubset(list(mods), k=k)

# ─────────────────────────────────────────────────────────────
# C. torchvision v2 增广管线
# ─────────────────────────────────────────────────────────────
class V2AugPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = v2.Compose([
            #v2.RandomHorizontalFlip(0.5),
            make_oneof(1,                                # 每次随机挑 5 个子增广
                v2.ColorJitter([0.5, 1.0], 1.0, 0.9, 0.1),

                 v2.RandomPerspective(distortion_scale=0.3, p=0.5),       # 额外给点透视扭曲
                 v2.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
                 v2.RandomApply(
                            [v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))],
                            p=0.8, ),
                # # 自定义增强（与您原先功能一一对应）
                 RandomSunFlare(p=1.0),
                 RandomCoarseDropout(num_holes=(2, 6), p=0.5),
                 RandomLEDStripe(p=0.3),
                 RandomLowLight(p=0.5),
                #
                # # torchvision v2 里本来就有的
                 v2.RandomPhotometricDistort(p=0.7),
                 v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            ),
            v2.Lambda(lambda t: t.clamp_(0, 1)),
        ])

    def forward(self, x):
        return self.aug(x)

# ─────────────────────────────────────────────────────────────
# D. 基准测试
# ─────────────────────────────────────────────────────────────
def bench(model, x, warm=10, repeat=50):
    for _ in range(warm):
        model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat


#
# from new_random import AugPipeline

import cv2
device =  "cuda"
img = cv2.cvtColor(cv2.imread("D:/Project/Scaling/lerobot-sim2real-main/lerobot-sim2real/greenscreen.png"), cv2.COLOR_BGR2RGB)
ten = torch.from_numpy(img).permute(2,0,1).float()/255
batch = ten.unsqueeze(0).repeat(888,1,1,1).to(device)

tv2_fast = V2AugPipeline().to(device)
# aug = AugPipeline().to( "cuda").eval()
# aug = torch.compile(aug, mode="reduce-overhead")
# traced_aug = torch.jit.trace(aug, batch[:1])
# traced_aug.save("aug_traced.pt")
# aug = torch.jit.load("aug_traced.pt").to(device).eval()
# out = loaded(batch)

import matplotlib.pyplot as plt
import time
s = time.time()
for i in range(50):
    t1 = time.time()
    with torch.no_grad():
        out = tv2_fast(batch)
    print("88 time cost", time.time()-t1)
    fig, axes = plt.subplots(3,3, figsize=(9,9))
    for i,ax in enumerate(axes.flat):
        ax.imshow((out[i]*255).byte().permute(1,2,0).cpu())
        ax.axis("off")
    plt.tight_layout(); plt.show()

print("end time",(time.time()-s)/50.0)

#
# if __name__ == "__main__":
#     device = "cuda"
#     B, C, H, W = 888, 3, 128, 128
#     x = torch.rand(B, C, H, W, device=device)
#
#     tv2_fast = V2AugPipeline().to(device)
#     print("tv2++:", bench(tv2_fast, x), "s / batch")
#
#     aug = AugPipeline().to("cuda").eval()
#     t1 = time.time()
#     for i in range(50):
#         out = aug(x)
#     print("time cost", (time.time() - t1)/50.0)