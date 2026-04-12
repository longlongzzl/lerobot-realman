import os
from typing import Dict, List, Union

import cv2
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import SimConfig


import cv2, albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

def deg2frac(deg):
    """任意角度（°）→ 0‑1 归一化比例"""
    return (deg % 360) / 360.0
ranges = [(deg2frac(-45), 1.0),   # 315°‑360°（0.875‑1.0）
          (0.0, deg2frac(45))]     #   0°‑45°（0‑0.125）



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

e
    Use `self.remove_object_from_grenscreen(object: Actor | Link | Articulation)` to exclude those objects from the greenscreen process.
    """

    rgb_overlay_paths: Dict[str, str] = None
    """dict mapping camera name to the file path of the greenscreening image"""
    _rgb_overlay_images: Dict[str, torch.Tensor] = dict()
    """dict mapping camera name to the image torch tensor"""
    rgb_overlay_mode: str = "background"
    """which RGB overlay mode to use during the greenscreen process. The default is 'background' which enables greenscreening like normal. The other option is 'debug' mode which
    will make the opacity of the original render and greenscreen overlay both 50%. The third option is "none" which will not perform any greenscreening."""

    _objects_to_remove_from_greenscreen: List[Union[Actor, Link]] = []
    """list of articulations/actors/links that should be removed from the greenscreen process"""
    _segmentation_ids_to_keep: torch.Tensor = None
    """torch tensor of segmentation ids that reference the objects that should not be greenscreened"""

    def __init__(self, **kwargs):
        # Load the "greenscreen" image, which is used to overlay the background portions of simulation observation
        if self.rgb_overlay_paths is not None:
            for camera_name, path in self.rgb_overlay_paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"rgb_overlay_path {path} is not found.")
                self._rgb_overlay_images[camera_name] = cv2.cvtColor(
                    cv2.imread(path), cv2.COLOR_BGR2RGB
                )  # (H, W, 3); float32
        else:
            self._rgb_overlay_images = None

        self.aug = A.Compose([
            # A.MotionBlur(blur_limit=7, p=0.4),
            A.OneOf([
                A.RandomSunFlare(src_radius=50, num_flare_circles_range=(2, 5),
                                 angle_range=(0.0, 0.125), p=1.0),
                A.PlanckianJitter(temperature_limit=(3000, 7000), p=1.0),
            ], p=0.7),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
            # ② 颜色抖动（加深饱和度/对比度）
            A.RandomBrightnessContrast(0.25, 0.4, p=0.7),
            A.HueSaturationValue(10, 40, 15, p=0.7),
            A.OneOf([
                A.RGBShift(r_shift_limit=20, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.ChannelShuffle(p=0.5)
            ], p=0.3),

            # ③ 噪声 / 模糊 / 压缩
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.GaussNoise(var_limit=(1, 5), noise_scale_factor=0.05, p=0.1),
                A.ImageCompression(quality_lower=30, quality_upper=70, p=0.1)
            ], p=0.5),

            # ④ 天气
            A.OneOf([
                # A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.05, p=0.1),
                A.RandomRain(blur_value=2, brightness_coefficient=0.9, p=0.1),
                A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=0.1)
            ], p=0.3),

            # ⑤ 几何
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
            #                    rotate_limit=10, border_mode=0, p=0.6),
            # A.HorizontalFlip(p=0.5),

            # ⑥ 遮挡
            A.CoarseDropout(num_holes_range=(2, 5),
                            hole_height_range=(8, 32),
                            hole_width_range=(8, 32),
                            fill=0, p=0.3),
            A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
            ToTensorV2()
        ])

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

    def _after_reconfigure(self, options: dict):
        super()._after_reconfigure(options)

        if self.rgb_overlay_mode != "none":
            # after reconfiguration in CPU/GPU sim we have initialized all ids of objects in the scene.
            # and can now get the list of segmentation ids to keep
            per_scene_ids = []
            for object in self._objects_to_remove_from_greenscreen:
                per_scene_ids.append(object.per_scene_id)
            self._segmentation_ids_to_keep = torch.unique(
                torch.concatenate(per_scene_ids)
            )

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

    def _green_sceen_rgb(self, rgb, segmentation, overlay_img):
        """returns green screened RGB data given a batch of RGB and segmentation images and one overlay image"""
        actor_seg = segmentation[..., 0]
        mask = torch.ones_like(actor_seg, device=actor_seg.device, dtype=torch.bool)
        if self._segmentation_ids_to_keep.device != actor_seg.device:
            self._segmentation_ids_to_keep = self._segmentation_ids_to_keep.to(
                actor_seg.device
            )
        if self.rgb_overlay_mode == "background":
            # only overlay the background and keep the foregrounds (robot and target objects) rendered in simulation
            mask[
                torch.isin(
                    actor_seg,
                    self._segmentation_ids_to_keep,
                )
            ] = 0
        mask = mask[..., None]

        # perform overlay on the RGB observation image
        if "debug" not in self.rgb_overlay_mode:
            rgb = rgb * (~mask) + overlay_img * mask
        else:
            rgb = rgb * 0.5 + overlay_img * 0.5
            rgb = rgb.to(torch.uint8)

        # img = rgb[:, :, ::-1].detach().cpu().numpy()
        # aug_img = self.aug(image=img)["image"]  # (C, H, W) tensor
        # rgb = aug_img.permute(1, 2, 0).to(torch.uint8)
        return rgb

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True):
        obs = super()._get_obs_sensor_data(apply_texture_transforms)

        # "greenscreen" process
        #assert 3 == 4
        if self.rgb_overlay_mode == "none":
            return obs

        #print(" self.obs_mode_struct.visual.rgb",  self.obs_mode_struct.visual.rgb,  self.obs_mode_struct.visual.segmentation, self.rgb_overlay_paths)
        #assert  1==2
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
                if (
                    self._rgb_overlay_images[camera_name].device
                    != obs[camera_name]["rgb"].device
                ):
                    self._rgb_overlay_images[camera_name] = self._rgb_overlay_images[
                        camera_name
                    ].to(obs[camera_name]["rgb"].device)
                overlay_img = self._rgb_overlay_images[camera_name]
                green_screened_rgb = self._green_sceen_rgb(
                    obs[camera_name]["rgb"],
                    obs[camera_name]["segmentation"],
                    overlay_img,
                )
                obs[camera_name]["rgb"] = green_screened_rgb
        return obs
