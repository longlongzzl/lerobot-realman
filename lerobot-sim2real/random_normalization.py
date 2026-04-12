import time

import cv2, albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

def deg2frac(deg):
    """任意角度（°）→ 0‑1 归一化比例"""
    return (deg % 360) / 360.0
ranges = [(deg2frac(-45), 1.0),   # 315°‑360°（0.875‑1.0）
          (0.0, deg2frac(45))]     #   0°‑45°（0‑0.125）

aug = A.Compose([
   # A.MotionBlur(blur_limit=7, p=0.4),
    A.OneOf([
        A.RandomSunFlare(src_radius=50, num_flare_circles_range=(2, 5),
                         angle_range=(0.0, 0.125), p=1.0),
        A.PlanckianJitter(temperature_limit=(3000, 7000), p=1.0),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
        # ② 颜色抖动（加深饱和度/对比度）
        A.RandomBrightnessContrast(0.25, 0.4, p=0.7),
        A.HueSaturationValue(10, 40, 15, p=0.7),
        A.RGBShift(r_shift_limit=20, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.ChannelShuffle(p=0.5),
        A.MotionBlur(blur_limit=3, p=0.3),
        A.GaussNoise(var_limit=(1, 5), noise_scale_factor=0.05, p=0.1),
        A.ImageCompression(quality_lower=30, quality_upper=70, p=0.1),
        A.RandomRain(blur_value=2, brightness_coefficient=0.9, p=0.1),
        A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=0.1),
        A.CoarseDropout(num_holes_range=(2, 5),
                        hole_height_range=(8, 32),
                        hole_width_range=(8, 32),
                        fill=0, p=0.3),
        A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
    ], p=0.7),



    # ⑤ 几何
    # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
    #                    rotate_limit=10, border_mode=0, p=0.6),
    #A.HorizontalFlip(p=0.5),

    # ⑥ 遮挡
    ToTensorV2()
])

# # 读入原图
img = cv2.imread(
    "D:/Project/Scaling/lerobot-sim2real-main/lerobot-sim2real/greenscreen.png"
)[:, :, ::-1]  # BGR → RGB



aug_img = aug(image=img)["image"]

for i in range(10000):
    # 生成并可视化 9 张增强图
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    t1 = time.time()
    for ax in axes.flatten():


        aug_img = aug(image=img)["image"]                     # (C, H, W) tensor
        img_np = aug_img.permute(1, 2, 0).cpu().numpy()       # 转 (H, W, C) numpy
        ax.imshow(img_np)
        ax.axis("off")
    print("Time cost", time.time() - t1 )
    plt.tight_layout()
    plt.show()
