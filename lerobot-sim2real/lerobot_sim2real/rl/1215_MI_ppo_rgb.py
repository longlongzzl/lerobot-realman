"""CleanRL Style PPO implementation for visual RL in ManiSkill. Taken from https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo

Only modification is Args is renamed to PPOArgs, the main function is put inside a train function for cross-module use, and we provide support to modify env kwargs
"""
from collections import defaultdict
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper, Flatten_Multi_RGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import torch.nn.functional as F

def _ensure_B1HW_bin(t: torch.Tensor) -> torch.Tensor:
    # 接受 [B,H,W] 或 [B,H,W,1]；返回 [B,1,H,W]，并转 float(0/1)
    if t.dim()==4 and t.shape[-1]==1: t = t.permute(0,3,1,2)
    elif t.dim()==3: t = t.unsqueeze(1)
    return (t>0).float()

def masks_from_infos(infos: dict, cam_idx: int):
    """
    从 infos 里拿当前帧二值 mask；你可把键名替换成自己环境里的实际名称。
    返回 [B,1,H,W] float(0/1) 或 None
    """
    # <<< 把下面候选键改成你实际的键名 >>>
    CAND_KEYS = [
        (f"mask_{cam_idx}",)                          # 单一目标
    ]
    for ks in CAND_KEYS:
        if all(k in infos for k in ks):
            m = 0
            for k in ks:
                m = m + _ensure_B1HW_bin(infos[k])
            return (m.clamp(0,1)).float()  # [B,1,H,W]
    return None


# def downsample_mask(m, h_attn, w_attn):
#     # m: [B,1,H,W] 二值
#     # 把每个目标像素“膨胀”到所在格子：等价于对二值图做 max 池化到目标分辨率
#     kh = math.ceil(m.shape[-2] / h_attn)
#     kw = math.ceil(m.shape[-1] / w_attn)
#     m2 = F.max_pool2d(m, kernel_size=(kh, kw), stride=(kh, kw), ceil_mode=True)
#     m2 = F.interpolate(m2, size=(h_attn, w_attn), mode="nearest")
#     return m2.clamp(0,1)
def downsample_mask(m, h_attn, w_attn):
    # m: [B,1,H,W] in {0,1}
    return F.adaptive_max_pool2d(m, (h_attn, w_attn))


def _find_raw_seg_from_obs(obs: dict, cam_idx: int):
    """
    从 obs 拿最原始 segmentation（整数标签），优先级：seg_* / segmentation_* / mask_* / seg / segmentation。
    返回 [B,H,W] (long) 或 None
    """
    cand = [f"mask_{cam_idx}"]
    for k in cand:
        if k in obs:
            t = obs[k]
            # 统一到 [B,H,W] long
            if t.dim()==4 and t.shape[-1]==1: t = t[...,0]
            if t.dim()==4 and t.shape[1]==1:  t = t[:,0]
            assert t.dim()==3, f"unexpected seg shape {t.shape} for key {k}"
            if not t.dtype.is_floating_point:
                return t.long()
            else:
                # 如果是浮点，先四舍五入到整数
                return t.round().long()
    return None

def _bin_from_raw_seg(seg_bhw: torch.Tensor) -> torch.Tensor:
    """raw seg -> 二值 [B,1,H,W] float(0/1)，只要 >0 就视为前景"""
    m = (seg_bhw > 0).float().unsqueeze(1)
    return m

def _colorize_labels(seg_bhw: torch.Tensor) -> torch.Tensor:
    """
    把整数标签上色，返回 [B,3,H,W] (0~1)。调色板固定、可复现。
    """
    B,H,W = seg_bhw.shape
    device = seg_bhw.device
    torch.manual_seed(0)
    palette = torch.randint(0, 256, (256,3), device=device, dtype=torch.int64)
    seg_mod = (seg_bhw % 256).clamp(min=0)
    # [B,H,W,3]
    col = palette[seg_mod]
    col = col.permute(0,3,1,2).float() / 255.0  # [B,3,H,W]
    # 背景(0)拉暗一点，避免太亮
    bg = (seg_bhw==0).unsqueeze(1)
    col = torch.where(bg, col*0.3, col)
    return col.clamp(0,1)

def _overlay_mask(rgb_chw: torch.Tensor, mask_b1hw: torch.Tensor, color_idx: int, alpha: float=0.6) -> torch.Tensor:
    """
    在 RGB 上叠加单通道 mask；color_idx: 0=R,1=G,2=B
    输入: rgb_chw [B,3,H,W] 0~1, mask [B,1,H,W] 0~1
    输出: [B,3,H,W] 0~1
    """
    color = torch.zeros_like(rgb_chw)
    color[:, color_idx:color_idx+1] = 1.0
    m = mask_b1hw.clamp(0,1)
    return (1 - alpha*m) * rgb_chw + (alpha*m) * color



@dataclass
class PPOArgs:

    search_entropy: int = 0
    act_type: int = 0
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""

    alpha: float = 0.0

    tcl_weight: float = 0.0

    sigma:float = 2.0
    method: str="softmax"
    end_alpha: float=1.0
    tcl_choice: int = 1
    "control the start"
    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_kwargs: dict = field(default_factory=dict)
    """extra environment kwargs to pass to the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = None #"pd_joint_delta_pos" #None
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25 # 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def contrastive_info_nce_single(
    image_representation: torch.Tensor,   # (N, D)
    private_state: torch.Tensor,          # (N, D)
    temperature: float = 0.2,
    normalize: bool = True,
    reduction: str = "mean",
):
    assert image_representation.shape == private_state.shape
    assert reduction in ("mean", "sum", "none")
    z = image_representation
    s = private_state
    if normalize:
        z = F.normalize(z, dim=1)
        s = F.normalize(s, dim=1)

    logits = (z @ s.t()) / temperature      # (N, N)
    labels = torch.arange(z.size(0), device=z.device)  # [0, 1, ..., N-1]
    loss = F.cross_entropy(logits, labels, reduction=reduction)
    return loss

class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

class AttnConvEncoderNCHW(nn.Module):
    """
    Drop-in 给 NatureCNN 的 rgb_* 提取器：
    输入:  x  [B, C, H, W]，数值范围 0~1（你的 NatureCNN 已经在前面 /255）
    输出:  z  [B, 256]  (与原先 fc 输出一致)
    额外:  self.last_attn  [B, 1, H', W']  注意力热图（前向即可拿）
    """
    def __init__(self, in_ch: int, feature_size: int = 256):
        super().__init__()
        # 与你原来完全一致的三层卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),    nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),    nn.ReLU(inplace=True),
        )
        # 1x1 注意力头 + Masked GAP -> 64d -> 256d
        self.attn_head = nn.Conv2d(64, 1, kernel_size=1)
        self.proj = nn.Sequential(nn.Linear(64, feature_size), nn.ReLU(inplace=True) )#,  nn.LayerNorm(feature_size))
        self.last_attn = None  # [B,1,H',W']

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        h = self.conv(x_nchw)                          # [B,64,H',W']
        logits = self.attn_head(h)
        A = torch.sigmoid(logits)           # [B,1,H',W']
        self.last_attn = A
        self.last_attn_logits = logits
        # Masked GAP，得到 [B,64]，再线性到 256
        w = A.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        z = (h * A).sum(dim=(2, 3)) / w.squeeze((2, 3))  # [B,64]

        z = self.proj(z)                               # [B,256]
        return z




class NatureCNN(nn.Module):
    def __init__(self, sample_obs, type="1"):
        super().__init__()

        assert isinstance(type, str)
        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb_0"].shape[-1]
        image_size=(sample_obs["rgb_0"].shape[1], sample_obs["rgb_0"].shape[2])

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here

        if type == "1" : # total share - 单目相机，只用rgb_0
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=int(in_channels),
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )

            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                n_flatten = cnn(sample_obs["rgb_0"].float().permute(0,3,1,2).cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors["rgb_0"] = nn.Sequential(cnn, fc)
            self.out_features += feature_size
        elif type == "2": # share but no fc share - 单目相机，只用rgb_0
            # here we use a NatureCNN architecture to process images, but any architecture is permissble here
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=int(in_channels),
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )

            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                n_flatten = cnn(sample_obs["rgb_0"].float().permute(0, 3, 1, 2).cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors["rgb_0"] = nn.Sequential(cnn, fc)
            self.out_features += feature_size

        elif type == "3":  # two cnn no share - 单目相机，只用rgb_0
            # here we use a NatureCNN architecture to process images, but any architecture is permissble here

            extractors["rgb_0"] = AttnConvEncoderNCHW(int(in_channels), feature_size)
            self.out_features += feature_size
        else:
            assert  1 == 2
      
        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Sequential(layer_init(nn.Linear(state_size, 256)),nn.ReLU(inplace=True), layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True)) #(nn.Sequential(layer_init(nn.Linear(state_size, 256)),nn.ReLU()))
            self.out_features += 256
        
        
       
        
        self.extractors = nn.ModuleDict(extractors)
        self.post_norm = nn.LayerNorm(self.out_features)



    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        key_latent_feature = {}
        attn_maps = {}  # <<< 新增：缓存每个相机的注意力
        attn_logits = {}
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if "rgb" in key:
                x = obs.float().permute(0, 3, 1, 2) / 255.0  # NCHW, 0~1
                z = extractor(x)  # [B,256]
                encoded_tensor_list.append(z)
                key_latent_feature[key] = z
                if hasattr(extractor, "last_attn") and extractor.last_attn is not None:
                    attn_maps[key] = extractor.last_attn  # [B,1,H',W']
                if hasattr(extractor, "last_attn_logits"):
                    attn_logits[key] = extractor.last_attn_logits
            elif key == "state":
                z = self.extractors["state"](obs)
                encoded_tensor_list.append(z)
                key_latent_feature[key] = z

        # <<< 新增：把注意力放到实例属性里，供 Agent 读取
        self.last_attn_maps = attn_maps
        self.last_attn_logits = attn_logits
        z_all = torch.cat(encoded_tensor_list, dim=1)
        #z_all = self.post_norm(z_all)
        return z_all, key_latent_feature
    #
    # def forward(self, observations) -> torch.Tensor:
    #     encoded_tensor_list = []
    #     # self.extractors contain nn.Modules that do all the processing.
    #     key_latent_feature = {}
    #     for key, extractor in self.extractors.items():
    #
    #         #print("????", observations)
    #         obs = observations[key]
    #         if "rgb" in key:
    #             obs = obs.float().permute(0,3,1,2)
    #             obs = obs / 255
    #         latent_feature = extractor(obs)
    #         encoded_tensor_list.append(latent_feature)
    #         key_latent_feature[key] = latent_feature
    #     return torch.cat(encoded_tensor_list, dim=1), key_latent_feature


def AvgL1Norm(x, eps=1e-8):
    return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)

import torch.nn.functional as F
import math
def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        #
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
        # Eq = F.softplus(q_samples) #+ q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        q_samples = torch.clamp(q_samples, -1e6, 9.5)

        # print("neg q samples ",q_samples.cpu().data.numpy())
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        assert 1 == 2

    if average:
        return Eq.mean()
    else:
        return Eq


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
        # Ep =  - F.softplus(-p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples

    elif measure == 'RKL':

        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        assert 1 == 2

    if average:
        return Ep.mean()
    else:
        return Ep


def fenchel_dual_loss(l, m, measure=None):
    '''Computes the f-divergence distance between positive and negative joint distributions.
    Note that vectors should be sent as 1x1.
    Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
    Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.
    Args:
        l: Local feature map.
        m: Multiple globals feature map.
        measure: f-divergence measure.
    Returns:
        torch.Tensor: Loss.
    '''
    N, units = l.size()

    # Outer product, we want a N x N x n_local x n_multi tensor.
    u = torch.mm(m, l.t())

    # Since we have a big tensor with both positive and negative samples, we need to mask.
    mask = torch.eye(N).to(l.device)
    n_mask = 1 - mask
    # Compute the positive and negative score. Average the spatial locations.
    E_pos = get_positive_expectation(u, measure, average=False)
    E_neg = get_negative_expectation(u, measure, average=False)
    MI = (E_pos * mask).sum(1)  # - (E_neg * n_mask).sum(1)/(N-1)
    # Mask positive and negative terms for positive and negative parts of loss
    E_pos_term = (E_pos * mask).sum(1)
    E_neg_term = (E_neg * n_mask).sum(1) / (N - 1)
    loss = E_neg_term - E_pos_term
    return loss, MI


def masked_huber_loss(y_pred, y_true, mask, delta=1.0):
    error = torch.abs(y_pred - y_true)
    loss = torch.where(error < delta, 0.5 * error ** 2, delta * (error - 0.5 * delta))
    masked_loss = loss * mask
    return masked_loss.sum() / (mask.sum() + 1e-6)



import copy
class Agent(nn.Module):
    def __init__(self, method, sigma, envs, sample_obs, net_type, choice, tcl_choice, device,z_detach, private_dim, search_entropy, act_type, critic_use_pi=0):
        super().__init__()
        self.z_detach = z_detach
        
        
        self.tcl_choice = tcl_choice
        self.last_attn_maps = {}
        self.feature_net = NatureCNN(sample_obs=sample_obs, type = net_type)
        self.device =device
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.choice = choice

        if choice in [1,  2]:
            self.z_dynamic_pre_1 = nn.Sequential(nn.Linear(256 + np.prod(envs.unwrapped.single_action_space.shape), 256), nn.ELU(), nn.Linear(256, 256),
                                               nn.ELU(), nn.Linear(256, 256))

            self.z_dynamic_pre_2 = nn.Sequential(nn.Linear(256 + np.prod(envs.unwrapped.single_action_space.shape), 256),
                                               nn.ELU(), nn.Linear(256, 256),
                                               nn.ELU(), nn.Linear(256, 256))

        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()

        self.critic_use_pi = critic_use_pi

        latent_size = self.feature_net.out_features

        if critic_use_pi :
            state_size = sample_obs["state"].shape[-1]
            self.critic_representation = nn.Sequential(
                layer_init(nn.Linear(state_size + private_dim, 512)),
                nn.ReLU(inplace=True))
            # self.policy_representation = nn.Sequential(
            #     layer_init(nn.Linear(latent_size, 256)),
            #     nn.Tanh())

        # self.actor_mean = nn.Sequential(
        #     layer_init(nn.Linear(latent_size, 256)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, 256)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        # )


        if act_type == 0 :
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(latent_size, 512)),
                nn.Tanh(),
                layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01 * np.sqrt(2)),
            )
        else:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(512, 256)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(256, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01 * np.sqrt(2)),
            )

        if search_entropy == 0:
            self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)
        elif search_entropy == 1:
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.unwrapped.single_action_space.shape)))  # * -0.5)


        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        #self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.unwrapped.single_action_space.shape)) ) #* -0.5)



        if choice == 4:
            self.nonlinearity = F.leaky_relu
            self.l1 = nn.Linear(256, 256)
            self.l2 = nn.Linear(256, 256)

            self.l3 = nn.Linear(256, 256)
            self.l4 = nn.Linear(256, 256)

        if choice == 5:
            self.nonlinearity = F.leaky_relu
            self.l1 = nn.Linear(256, 256)
            self.l2 = nn.Linear(256, 256)

            self.l3 = nn.Linear(256, 256)
            self.l4 = nn.Linear(256, 256)

        if choice == 3:
            self.l1 = nn.Sequential(
                nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 256),
                nn.ELU(), nn.Linear(256, private_dim))
        self.private_dim = private_dim
        if choice == 6:
            self.W1 = nn.Parameter(torch.rand(256, 256))
            self.W2 = nn.Parameter(torch.rand(256, 256))
        if choice == 7:
            self.W1 = nn.Parameter(torch.rand(256, private_dim))
            self.W2 = nn.Parameter(torch.rand(256, private_dim))
        if choice == 8:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(self.private_dim, 256)),nn.ReLU(inplace=True), layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True))
        if choice == 9:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(self.private_dim, 256)), nn.ReLU(inplace=True))




        if choice == 10:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(64, 256)), nn.ReLU(inplace=True),
                                                  layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True))
        if choice == 11:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(128, 256)), nn.ReLU(inplace=True),
                                                  layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True))
        if choice == 12:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True),
                                                  layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True))

        if choice == 13:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(32, 256)), nn.ReLU(inplace=True),
                                                  layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True))
        if choice == 14:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(16, 256)), nn.ReLU(inplace=True),
                                                  layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True))

        if choice == 101:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(64, 256)), nn.ReLU(inplace=True),
                                                  layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True))
        elif choice == 102:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(64, 256)), nn.ReLU(inplace=True),
                                                  layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True),
                                                  layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True))
        elif choice == 103:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(64, 256)), nn.ReLU(inplace=True),
                                                  layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True),
                                                  layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True))
        elif choice == 104:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(64, 256)), nn.ReLU(inplace=True),
                                                  layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True))
        elif choice == 105:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(64, 256)), nn.ReLU(inplace=True),
                                                  layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True))
        elif choice == 106:
            self.private_state_fc = nn.Sequential(layer_init(nn.Linear(64, 256)), nn.ReLU(inplace=True))

        temp_out_dim = 64



        # 单目相机：输入维度从512改为256（只用rgb_0）
        self.img_proj = nn.Sequential(
            layer_init(nn.Linear(256, 256)), nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, temp_out_dim)),
        )


        if choice == 101:
            self.state_proj = nn.Sequential(
                layer_init(nn.Linear(self.private_dim, 128)), nn.ReLU(inplace=True),
                layer_init(nn.Linear(128, 128)), nn.ReLU(inplace=True),
                layer_init(nn.Linear(128, temp_out_dim)),
            )
        elif choice == 103:
            self.state_proj = nn.Sequential(
                layer_init(nn.Linear(self.private_dim, 128)), nn.ReLU(inplace=True),
                layer_init(nn.Linear(128, 128)), nn.ReLU(inplace=True),
                layer_init(nn.Linear(128, temp_out_dim)),
            )
        elif choice == 104:
            self.state_proj = nn.Sequential(
                layer_init(nn.Linear(self.private_dim, temp_out_dim)),
            )
        elif choice == 105:
            self.state_proj = nn.Sequential(
                layer_init(nn.Linear(self.private_dim, 256)), nn.ReLU(inplace=True),
                layer_init(nn.Linear(256, temp_out_dim)),
            )
        else:
            self.state_proj = nn.Sequential(
                layer_init(nn.Linear(self.private_dim, 128)), nn.ReLU(inplace=True),
                layer_init(nn.Linear(128, temp_out_dim)),
            )
        self.bt_lambda = 5e-3  # Barlow Twins off-diagonal 权重，论文默认 ~5e-3
        self.tau = 0.07      # InfoNCE 温度

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / getattr(self, "tau", 0.07))))
        self.logit_scale_min = math.log(1 / 100)
        self.logit_scale_max = math.log(100)

        self.sigma= sigma
        self.method = method
        # self.image_fc =  nn.Sequential(layer_init(nn.Linear(256, self.private_dim)),nn.ReLU(inplace=True))

    def _standardize(self, z, eps=1e-4):
        # 按批对每个维度做零均值/单位方差标准化（RL小批次比BN更稳）
        z = z - z.mean(dim=0, keepdim=True)
        z = z / (z.std(dim=0, keepdim=True) + eps)
        return z

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def barlow_twins_loss(self, z_o, z_s):
        # z_o, z_s: [B, D]
        B, D = z_o.size(0), z_o.size(1)
        z_o = self._standardize(z_o)
        z_s = self._standardize(z_s)
        c = (z_o.T @ z_s) / B  # [D, D] 跨相关矩阵

        diag = c.diag()
        off_vals = self._off_diagonal(c)
        on_diag = (diag - 1).pow(2).mean()
        off_diag = off_vals.pow(2).mean()
        loss = on_diag + self.bt_lambda * off_diag
        with torch.no_grad():
            diag_mean = diag.mean()
            diag_std = diag.std(unbiased=False)
            off_diag_mean = off_vals.abs().mean()  # 仅非对角的平均
            I = torch.eye(D, device=c.device, dtype=c.dtype)
            fro_norm = torch.norm(c - I, p='fro')


        return loss, {"diag_mean": diag_mean,"diag_std": diag_std,"off_diag_mean": off_diag_mean, "fro_norm": fro_norm}
        # 参见 Barlow Twins 原式。:contentReference[oaicite:1]{index=1}

    def siglip_bce_loss(self, q, k, pos_w=None, tau=None, symmetric=True):
        # q, k: [B, D] 已是同空间的嵌入
        q = F.normalize(q, dim=1, eps=1e-8)
        k = F.normalize(k, dim=1, eps=1e-8)
        if tau is None:
            tau = self.tau  # 你原来的温度，等价于一个logit scale
        logits = (q @ k.T) / tau  # [B, B]
        B = logits.size(0)

        # 逐元素二分类：对角=1，非对角=0
        target = torch.zeros_like(logits)
        target.fill_(0.0)
        target.fill_diagonal_(1.0)

        # 正负极度不平衡，给正样本更大权重（建议 pos_w=B-1）
        if pos_w is None:
            pos_w = float(B - 1)
        w = torch.ones_like(logits)
        w.fill_(1.0)
        w.fill_diagonal_(pos_w)

        loss_f = F.binary_cross_entropy_with_logits(logits, target, weight=w, reduction='mean')

        if symmetric:
            loss_b = F.binary_cross_entropy_with_logits(logits.T, target.T, weight=w.T, reduction='mean')
            loss = 0.5 * (loss_f + loss_b)
        else:
            loss = loss_f

        # 统计项（可选）
        with torch.no_grad():
            sim = (logits * tau)
            pos_sim_mean = sim.diag().mean()
            mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
            hard_neg_sim = sim.masked_select(mask).view(B, B - 1).max(dim=1).values.mean()
        return loss, {"pos_sim_mean": pos_sim_mean, "hard_neg_sim": hard_neg_sim}



    def info_nce_loss(self, q, k):
        # 对称 InfoNCE（q->k 与 k->q）
        q = F.normalize(q, dim=1, eps=1e-8)
        k = F.normalize(k, dim=1, eps=1e-8)

        logits = (q @ k.T) / self.tau  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_f = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        loss = 0.5 * (loss_f + loss_b)

        with torch.no_grad():
            B = logits.size(0)
            sim = logits * self.tau  # 还原为余弦相似度
            pos_sim_mean = sim.diag().mean()
            mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
            hard_neg_sim = sim.masked_select(mask).view(B, B - 1).max(dim=1).values.mean()
            top1_acc_row = (logits.argmax(dim=1) == labels).float().mean()
            top1_acc_col = (logits.argmax(dim=0) == labels).float().mean()

        return loss, {
            "pos_sim_mean": pos_sim_mean,
            "hard_neg_sim": hard_neg_sim,
            "top1_acc_row": top1_acc_row,
            "top1_acc_col": top1_acc_col,
        }

    def contrastive_with_adv_weights(self, q, k, D, adv):
        """
        q,k: [B,D]  投影后的表示
        D:   [B,B]  L2距离矩阵(对角=0)
        adv: [B]    PPO/GAE 优势
        返回: (loss, stats)
        """
        # ==== 内置稳妥默认 ====
        tau = getattr(self, "tau", 0.2)  # InfoNCE温度（CLIP/SimCLR常用）  [Radford21; SimCLR]
        beta_dist = 1.0  # 距离温度：远→重（如需近→重，把 sign 设 -1.0）
        beta_adv = 1.0  # 优势温度：优势大→样本更重  [AWR 思路]
        lambda_reg = 1.0  # 负样重分配正则强度

        # ==== 早退 ====
        B = q.size(0)
        if B < 2:
            return q.new_zeros(()), {"note": "batch too small"}

        # ==== 相似度 + 对称 log-softmax（CLIP式）====
        q = F.normalize(q, dim=1, eps=1e-8)
        k = F.normalize(k, dim=1, eps=1e-8)
        s = (q @ k.T) / tau
        eye = torch.eye(B, device=s.device, dtype=torch.bool)
        logp_row = s - torch.logsumexp(s, dim=1, keepdim=True)  # [B,B]
        logp_col = s.T - torch.logsumexp(s.T, dim=1, keepdim=True)  # [B,B]

        # ==== 硬标签CE：强对齐（对角 one-hot）====
        ce_i2t = -logp_row.diag().mean()
        ce_t2i = -logp_col.diag().mean()

        # ==== 距离→负样软标签（只在 j!=i 上归一；远→重）====
        with torch.no_grad():
            D = D.detach()
            off = D[~eye]
            p95 = torch.quantile(off, 0.95) if off.numel() > 0 else torch.tensor(1.0, device=D.device)
            Dn = D / p95.clamp_min(1e-6)
        sign = +1.0  # 远→重；若想近→重设 -1.0
        w_row = torch.softmax(sign * beta_dist * Dn.masked_fill(eye, float('-inf')), dim=1)  # 行：只负样
        w_col = torch.softmax(sign * beta_dist * Dn.masked_fill(eye, float('-inf')), dim=0)  # 列：只负样

        # ==== 优势→样本级配额（不反传到权重）====
        with torch.no_grad():
            a = (adv.detach() - adv.mean()) / (adv.std(unbiased=False) + 1e-12)
            w_adv = torch.softmax(beta_adv * a, dim=0)  # [B]

        # ==== 负样加权正则（软标签CE）× 优势样本权重 ====
        reg_i2t = (w_adv * (-(w_row * logp_row).sum(dim=1))).mean()
        reg_t2i = (w_adv * (-(w_col * logp_col).sum(dim=1))).mean()

        loss = 0.5 * ((ce_i2t + lambda_reg * reg_i2t) + (ce_t2i + lambda_reg * reg_t2i))

        with torch.no_grad():
            stats = {
                "top1_i2t": (s.argmax(1) == torch.arange(B, device=s.device)).float().mean().item(),
                "top1_t2i": (s.T.argmax(1) == torch.arange(B, device=s.device)).float().mean().item(),
                "w_adv_max": w_adv.max().item(),
                "w_row_p95": torch.quantile(w_row[~eye], 0.95).item() if (~eye).any() else 0.0,
                "w_col_p95": torch.quantile(w_col[~eye], 0.95).item() if (~eye).any() else 0.0,
            }
        return loss, stats


    def pure_adv_info_nce_loss(self, q, k, adv=None,  adv_tau=1.0, method="softmax"):
        """
        Advantage-Weighted InfoNCE for PPO.
        q: [B, D]  (image-side projection)
        k: [B, D]  (state/privileged-side projection)
        adv: [B]   (per-sample advantages; unnormalized is fine)
        sigma:  controls how sharply advantages reweight samples
        """
        # 1) normalize features
        q = F.normalize(q, dim=1, eps=1e-8)
        k = F.normalize(k, dim=1, eps=1e-8)
        # 2) learnable temperature (fallback to 1/tau if not set)

        # if use_logits:
        #     # clamp logit_scale (in log space) to avoid overflow/underflow
        #     logit_scale = self.logit_scale.clamp(self.logit_scale_min, self.logit_scale_max).exp()
        #     logits = (q @ k.T) * logit_scale
        # else:
        logits = (q @ k.T) / self.tau
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        w = adv

        loss_vec = F.cross_entropy(logits, labels, reduction="none")
        loss_vec_b = F.cross_entropy(logits.T, labels, reduction="none")

        if w is None:
            weight = None
            loss_f = loss_vec.mean()
            loss_b = loss_vec_b.mean()
        else:
            if method == "softmax":
                weight = B* F.softmax(w.detach() / adv_tau, dim=0)
            elif method == "sigmod":
                p = torch.sigmoid(w.detach() / adv_tau)
                weight = p / p.mean()
            elif method == "exp":
                weight = torch.exp(adv / adv_tau)
                weight = weight / weight.mean()  # 归一化到均值=1，而非和=B
            else:
                weight = None
            loss_f = ( weight * loss_vec).mean()
            loss_b = ( weight * loss_vec_b).mean()
        loss = 0.5 * (loss_f + loss_b)

        # 7) logging
        with torch.no_grad():
            sim = (q @ k.T)  # 余弦相似度（已归一化，无需乘/除温度）
            pos_sim_mean = sim.diag().mean()
            mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
            hard_neg_sim = sim.masked_select(mask).view(B, B - 1).max(dim=1).values.mean()
            top1_acc_row = (logits.argmax(dim=1) == labels).float().mean()
            top1_acc_col = (logits.argmax(dim=0) == labels).float().mean()
        # logit_scale_val = logit_scale.detach() if torch.is_tensor(logit_scale) else torch.tensor(logit_scale)

        if weight is not None:
            weight_max = weight.max()
            weight_mean = weight.mean()
            weight_min = weight.min()
        else:
            weight_max = torch.tensor(0.0, device=logits.device)
            weight_mean = torch.tensor(0.0, device=logits.device)
            weight_min = torch.tensor(0.0, device=logits.device)

        return loss, {
            "weight_max": weight_max,
            "weight_mean": weight_mean ,
            "weight_min": weight_min,
            "pos_sim_mean": pos_sim_mean,
            "hard_neg_sim": hard_neg_sim,
            "top1_acc_row": top1_acc_row,
            "top1_acc_col": top1_acc_col,
            "aw_used": torch.tensor(float(adv is not None), device=logits.device),
            "logit_scale": self.logit_scale
        }




    def pair_wise_adv_info_nce_loss(
            self, q, k, adv=None, label_smoothing=0.05,
            lambda_w=0.2,
    ):
        # 1) normalize features
        q = F.normalize(q, dim=1, eps=1e-8)
        k = F.normalize(k, dim=1, eps=1e-8)

        # 2) logits with temperature or learnable scale
        logits = (q @ k.T) / self.tau

        B = logits.size(0)
        device = logits.device
        labels = torch.arange(B, device=device)
        eye = torch.eye(B, device=device, dtype=torch.bool)

        # === 没有优势 or 优势几乎无差别 → 退化为原始对称 InfoNCE ===
        if (adv is None) or (adv.std(unbiased=False) < 1e-6):
            loss_f = F.cross_entropy(logits, labels, reduction="none", label_smoothing=label_smoothing).mean()
            loss_b = F.cross_entropy(logits.T, labels, reduction="none", label_smoothing=label_smoothing).mean()
            loss = 0.5 * (loss_f + loss_b)
            with torch.no_grad():
                sim = (q @ k.T)
                pos_sim_mean = sim.diag().mean()
                mask = ~eye
                hard_neg_sim = sim.masked_select(mask).view(B, B - 1).max(dim=1).values.mean()
                top1_acc_row = (logits.argmax(dim=1) == labels).float().mean()
                top1_acc_col = (logits.argmax(dim=0) == labels).float().mean()
            return loss, {
                "pos_sim_mean": pos_sim_mean,
                "hard_neg_sim": hard_neg_sim,
                "top1_acc_row": top1_acc_row,
                "top1_acc_col": top1_acc_col,
                "mean_alpha_neg": torch.tensor(float('nan'), device=device),
                "mean_delta_neg": torch.tensor(0.0, device=device),
                "aw_used": torch.tensor(0.0, device=device),
                "lambda_w": torch.tensor(float(lambda_w), device=device),
                "logit_scale": getattr(self, "logit_scale", None),
            }

        # === 基于“优势真值”的 pair-wise 权重（最小改动版） ===
        with torch.no_grad():
            # >>> CHANGED: 先做 z-score + 温度，防 sigmoid 饱和
            z = adv
            p = torch.sigmoid(z)  # <<< 改这里

            # 成对差异 Δ_ij = |p_i - p_j|，对角 0
            pi, pj = p.view(B, 1), p.view(1, B)
            Delta = (pi - pj).abs().masked_fill(eye, 0.0)

            # 原始负样本权重（只在非对角位置）
            alpha_neg_raw = Delta + 1e-12

            # >>> CHANGED: 不再做“行归一 + 夹断 + 再归一”
            #              换成 log 域“行内零均值 + 截断”（等价于除几何均值 + 限幅）
            log_alpha = torch.log(alpha_neg_raw)
            log_alpha_centered = log_alpha - log_alpha.mean(1, keepdim=True)  # 行均值=0
            c = 1.0  # <<< 限幅；可在 0.5~1.5 间调
            shift = log_alpha_centered.clamp(-c, c)  # [-c, c]
            shift.masked_fill_(eye, 0.0)  # 对角不动（正样本）

            # 为了兼容你原有的 logging 字段（可选）
            alpha_centered_bounded = torch.exp(shift)  # 行几何均值≈1 的“等效 α”
            alpha_neg = alpha_centered_bounded.clone()  # 占位，供后面 mean_alpha_neg 统计
            # <<< 到此为止，改动仅限于这几行
        # === 在 logit 里加 λ_w * 行内零均值后的移位（只影响负样本） ===
        logits_w = logits + lambda_w * shift

        # 对称 CE（行 & 列）
        loss_f = F.cross_entropy(logits_w, labels, reduction="none", label_smoothing=label_smoothing).mean()
        loss_b = F.cross_entropy(logits_w.T, labels, reduction="none", label_smoothing=label_smoothing).mean()
        loss = 0.5 * (loss_f + loss_b)

        # logging（尽量保持你原有键值）
        with torch.no_grad():
            sim = (q @ k.T)
            pos_sim_mean = sim.diag().mean()
            mask = ~eye
            hard_neg_sim = sim.masked_select(mask).view(B, B - 1).max(dim=1).values.mean()
            top1_acc_row = (logits.argmax(dim=1) == labels).float().mean()  # 仍用未加权 logits（与你原来一致）
            top1_acc_col = (logits.argmax(dim=0) == labels).float().mean()
            mean_alpha_neg = alpha_neg.masked_select(~eye).mean()  # ≈ 1（行几何均值≈1）
            mean_delta_neg = Delta.mean()

            logits_mean = logits.mean()
            logits_min = logits.min()
            logits_max = logits.max()

            # 2) shift（你加到 logits 上的偏置；只看非对角，因为对角我们设为 0）
            shift_nd = shift.masked_select(~eye)  # non-diagonal entries
            shift_mean = shift_nd.mean()
            shift_min = shift_nd.min()
            shift_max = shift_nd.max()

            # 3) 可选：实际生效的移位（lambda_w * shift），量纲与 logits 一致，便于判断强弱
            eff_shift_nd = (lambda_w * shift_nd)
            eff_shift_mean = eff_shift_nd.mean()
            eff_shift_min = eff_shift_nd.min()
            eff_shift_max = eff_shift_nd.max()


        return loss,\
               {"logits_mean": logits_mean,
            "logits_min": logits_min,
            "logits_max": logits_max,
            "shift_mean": shift_mean,
            "shift_min": shift_min,
            "shift_max": shift_max,
            # 可选：更直观的“有效移位”（含 lambda_w）
            "eff_shift_mean": eff_shift_mean,
            "eff_shift_min": eff_shift_min,
            "eff_shift_max": eff_shift_max,
            "adv_mean": adv.mean(),
            "adv_min": adv.min(),
            "adv_max": adv.max(),
            "sigmod_adv_mean": p.mean(),
            "sigmod_adv_min": p.min(),
            "sigmod_adv_max": p.max(),
            "pos_sim_mean": pos_sim_mean,
            "hard_neg_sim": hard_neg_sim,
            "top1_acc_row": top1_acc_row,
            "top1_acc_col": top1_acc_col,
            "mean_alpha_neg": mean_alpha_neg,
            "mean_delta_neg": mean_delta_neg,
            "lambda_w": torch.tensor(float(lambda_w), device=device),
            "alpha_clip": torch.tensor(float(c), device=device),  # 方便你记录
        }

    def adv_info_nce_loss(self, q, k, adv=None, use_logits=False, sigma=2.0, label_smoothing=0.05, new_tau=None):
        """
        Advantage-Weighted InfoNCE for PPO.
        q: [B, D]  (image-side projection)
        k: [B, D]  (state/privileged-side projection)
        adv: [B]   (per-sample advantages; unnormalized is fine)
        sigma:  controls how sharply advantages reweight samples
        """
        # 1) normalize features
        q = F.normalize(q, dim=1, eps=1e-8)
        k = F.normalize(k, dim=1, eps=1e-8)
        # 2) learnable temperature (fallback to 1/tau if not set)

        if new_tau is None:
            new_tau = self.tau

        if use_logits:
            # clamp logit_scale (in log space) to avoid overflow/underflow
            logit_scale = self.logit_scale.clamp(self.logit_scale_min, self.logit_scale_max).exp()
            logits = (q @ k.T) * logit_scale
        else:
            logits = (q @ k.T) / new_tau
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)


#        if adv is None:
#            w = None
#        else:
#            with torch.no_grad():
#                w = F.softplus(adv / sigma)  # 非负，弱化负优势
#                w = w / (w.mean() + 1e-8)  # 归一化到均值=1，避免缩放学习率
        w = None
        
        loss_vec = F.cross_entropy(logits, labels, reduction="none", label_smoothing=label_smoothing)
        loss_f = (w * loss_vec).mean() if w is not None else loss_vec.mean()

        loss_vec_b = F.cross_entropy(logits.T, labels, reduction="none", label_smoothing=label_smoothing)
        loss_b = (w * loss_vec_b).mean() if w is not None else loss_vec_b.mean()
        loss = 0.5 * (loss_f + loss_b)

        # 7) logging
        with torch.no_grad():
            sim = (q @ k.T)  # 余弦相似度（已归一化，无需乘/除温度）
            pos_sim_mean = sim.diag().mean()
            mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
            hard_neg_sim = sim.masked_select(mask).view(B, B - 1).max(dim=1).values.mean()
            top1_acc_row = (logits.argmax(dim=1) == labels).float().mean()
            top1_acc_col = (logits.argmax(dim=0) == labels).float().mean()
           # logit_scale_val = logit_scale.detach() if torch.is_tensor(logit_scale) else torch.tensor(logit_scale)

        return loss, {
            "pos_sim_mean": pos_sim_mean,
            "hard_neg_sim": hard_neg_sim,
            "top1_acc_row": top1_acc_row,
            "top1_acc_col": top1_acc_col,
            "aw_sigma": torch.tensor(float(sigma), device=logits.device),
            "aw_used": torch.tensor(float(adv is not None), device=logits.device),
            "logit_scale": self.logit_scale
        }

    def compute_logits(self, z_a, z_pos, type=0):
        if type == 0 :
            W = self.W1
        else:
            W = self.W2
        Wz = torch.matmul(W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


    def get_features(self, x):
        return self.feature_net(x)[0]
    def get_value(self, x, private_state):

        if self.critic_use_pi:
        
           # print("???", x["state"].shape, private_state.shape)
            z = self.critic_representation(torch.cat([x["state"], private_state], -1))
            value = self.critic(z)

        else:
            x = self.feature_net(x)[0]
            value = self.critic(x)
        return value
    def get_action(self, x, private_state, deterministic=False, alpha = 0.0):
        x, feature_dict = self.feature_net(x)
        #if self.choice == 8:
            #private_state = self.private_state_fc(private_state)



        # 单目相机：只用rgb_0
        image_representation = feature_dict["rgb_0"]
        
        # image_representation = self.image_fc(image_representation)

        img_rep = self.img_proj(image_representation)
        z = self.private_state_fc(img_rep)
        x = torch.cat([feature_dict["state"], z], -1)

        #x = self.policy_representation(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def info_nce_weighted_by_distance(self, q, k, D,beta=1.0):
        """
        q, k: [B, D]  已做投影头; 本函数内部会做 L2 归一化
        D: [B, B]     L2 距离矩阵（对角为0）
        tau:          InfoNCE 温度（建议 0.2 起步）
        beta:         距离→权重的温度倒数；这里用 +beta*D（远→权重大）
        lam:          本项在总损失中的系数（外层还可再做调度）
        """
        # 1) 基础相似度（与 CLIP/SimCLR 一致）
        q = F.normalize(q, dim=1, eps=1e-8)
        k = F.normalize(k, dim=1, eps=1e-8)
        s = (q @ k.T) / self.tau  # [B,B]
        B = s.size(0)
        device = s.device
        eye = torch.eye(B, device=device, dtype=torch.bool)

        # === 稳尺度：先做一个鲁棒缩放，避免softmax被极端距离吃满 ===
        with torch.no_grad():
            p95 = torch.quantile(D[~eye], 0.95).clamp_min(1e-6)
        Dn = D / p95

        # 2) 行向权重（i->t）：softmax(+beta * D) 远→重；对角置0后行归一
        w_row = torch.softmax(beta * Dn.masked_fill(eye, 0.0), dim=1)  # 每行和=1

        # 3) 列向权重（t->i）：对应列softmax，做列归一
        w_col = torch.softmax(beta * Dn.masked_fill(eye, 0.0), dim=0)  # 每列和=1

        # 4) 加权 InfoNCE（权重在损失外，不改 softmax 概率）
        # 行 softmax 的 log 概率：每行 i 对所有 j
        logp_row = s - torch.logsumexp(s, dim=1, keepdim=True)  # [B,B]
        loss_i2t = -(w_row * logp_row).sum(dim=1).mean()

        # 列 softmax 的 log 概率（通过转置实现列 softmax）
        logp_col = s.T - torch.logsumexp(s.T, dim=1, keepdim=True)  # [B,B]
        loss_t2i = -(w_col * logp_col).sum(dim=1).mean()

        loss = 0.5 * (loss_i2t + loss_t2i)

        # ——精简监控（够看就行）——
        with torch.no_grad():
            monitor = {
                "top1_i2t": (s.argmax(1) == torch.arange(B, device=device)).float().mean(),
                "top1_t2i": (s.T.argmax(1) == torch.arange(B, device=device)).float().mean(),
                "w_row_p95": torch.quantile(w_row[~eye], 0.95),
                "w_col_p95": torch.quantile(w_col[~eye], 0.95),
            }
        return loss, monitor

    def S_from_distance_linear(self, D, lo_p=0.05, hi_p=0.95):
        """
        D: [B,B] 非负距离矩阵（对角建议=0）
        返回 S ∈ [-1,1]，对角=1；距离越小，相似度越大
        """
        B = D.size(0)
        device = D.device
        eye = torch.eye(B, device=device, dtype=torch.bool)

        # 稳健 min-max 到 [0,1]
        off = D[~eye]
        if lo_p is None or hi_p is None:
            lo, hi = off.min(), off.max()
        else:
            lo = torch.quantile(off, lo_p)
            hi = torch.quantile(off, hi_p)
        scale = (hi - lo).clamp_min(1e-12)
        D01 = ((D - lo) / scale).clamp(0.0, 1.0)

        # 距离→相似度，再映射 [-1,1]
        S = (1.0 - D01) * 2.0 - 1.0
        S = 0.5 * (S + S.T)  # 保证对称
        S = S.masked_fill(eye, 1.0)  # 对角置 1
        return S

    def S_from_adv_pairwise(self,adv, lo_p=0.05, hi_p=0.95):
        """
        adv: [B]
        返回 S_adv ∈ [-1,1] 的对称矩阵；任一方 adv 越大，该对值越小
        """
        B = adv.numel()
        device = adv.device
        eye = torch.eye(B, device=device, dtype=torch.bool)

        # 先把 adv 线性缩放到 [0,1]
        x = adv
        if lo_p is None or hi_p is None:
            lo, hi = x.min(), x.max()
        else:
            lo = torch.quantile(x, lo_p)
            hi = torch.quantile(x, hi_p)
        scale = (hi - lo).clamp_min(1e-12)
        a01 = ((x - lo) / scale).clamp(0.0, 1.0)  # adv 大→a01 大

        # 成对聚合：对 (i,j) 用 max(a01_i, a01_j)
        M = torch.maximum(a01[:, None], a01[None, :])  # [B,B] ∈ [0,1]
        S = (1.0 - M) * 2.0 - 1.0  # 映射到 [-1,1]
        S = 0.5 * (S + S.T)
        S = S.masked_fill(eye, 1.0)
        return S


    def cosine_similarity_matrix(self,private_obs, eps=1e-8):
        # private_obs: [B, D]
        x = F.normalize(private_obs, dim=1, eps=eps)  # 逐样本 L2 归一化
        S = x @ x.T  # [B, B], ∈ [-1, 1]
        return S

    def l2_distance_matrix(self, private_obs: torch.Tensor,  eps: float = 1e-12):
        """
        private_obs: [B, D]
        返回: [B, B] 的 L2 距离矩阵（或平方距离矩阵）
        """
        X = private_obs  # [B, D]
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
        x2 = (X * X).sum(dim=1, keepdim=True)  # [B, 1]
        d2 = (x2 + x2.T - 2.0 * (X @ X.T)).clamp_min(0.0)  # 数值稳定
        return (d2 + eps).sqrt()

    def info_nce_weighted_by_advpair(self, q, k, adv, beta: float = 1.0, lambda_reg: float = 1.0):

        B = q.size(0)
        # 1) 相似度（CLIP/SimCLR），tau 用 self.tau
        q = F.normalize(q, dim=1, eps=1e-8)
        k = F.normalize(k, dim=1, eps=1e-8)
        s = (q @ k.T) / self.tau  # [B,B]
        eye = torch.eye(B, device=s.device, dtype=torch.bool)

        # 行/列 log-softmax 概率（不改 logits 概率，只在损失外加权）
        logp_row = s - torch.logsumexp(s, dim=1, keepdim=True)  # [B,B]
        logp_col = s.T - torch.logsumexp(s.T, dim=1, keepdim=True)  # [B,B]

        # 2) 正样硬标签 CE（标准 InfoNCE/NT-Xent 主项）
        ce_i2t = -logp_row.diag().mean()
        ce_t2i = -logp_col.diag().mean()

        if adv is None:
            loss = 0.5 * (ce_i2t + ce_t2i)
            with torch.no_grad():
                stats = {
                    "top1_i2t": (s.argmax(1) == torch.arange(B, device=s.device)).float().mean(),
                    "top1_t2i": (s.T.argmax(1) == torch.arange(B, device=s.device)).float().mean(),
                }
            return loss, stats

        # 3) 用“优势差”构造配对级软标签：|A_i - A_j|
        with torch.no_grad():
            A = adv.detach().view(B)
            pair = (A[:, None] - A[None, :]).abs()  # [B,B], 对角=0
            off = pair[~eye]
            # 鲁棒缩放，防极端值把 softmax 吃满
            p95 = torch.quantile(off, 0.95) if off.numel() > 0 else torch.tensor(1.0, device=pair.device)
            pair_n = pair / p95.clamp_min(1e-6)

        # 只在负样上归一：对角置 -inf
        w_row = torch.softmax(beta * pair_n.masked_fill(eye, float('-inf')), dim=1)  # i→t
        w_col = torch.softmax(beta * pair_n.masked_fill(eye, float('-inf')), dim=0)  # t→i
        # 4) 负样软标签 CE（损失外）+ 正样硬 CE（合并）
        reg_i2t = -(w_row * logp_row).sum(dim=1).mean()
        reg_t2i = -(w_col * logp_col).sum(dim=1).mean()
        loss = 0.5 * ((ce_i2t + lambda_reg * reg_i2t) +
                      (ce_t2i + lambda_reg * reg_t2i))

        # 5) 简洁监控
        with torch.no_grad():
            stats = {
                "top1_i2t": (s.argmax(1) == torch.arange(B, device=s.device)).float().mean(),
                "top1_t2i": (s.T.argmax(1) == torch.arange(B, device=s.device)).float().mean(),
            }
        return loss, stats

    # def info_nce_weighted_by_advpair(self, q, k, adv, beta: float = 1.0):
    #     """
    #     仅配对级（pair-level）软标签的加权 InfoNCE（对称版）：
    #       - 软标签来自 |GAE_i - GAE_j|：差越大 → 权重越大（beta>0）
    #       - 不做样本级权重；不改 logits，权重在损失外（软标签CE）
    #     参数
    #       q, k: [B, D] 投影后的表示（函数内会 L2 归一）
    #       adv:  [B]    本批 GAE（你已 z-score 更好；这里默认直接用）
    #       beta:        配对温度；>0 强调“差大→更重”；<0 切到“差小→更重”
    #     返回
    #       loss:  标量
    #       stats: 监控字典（简版）
    #     """
    #     B = q.size(0)
    #     if B < 2:
    #         return q.new_zeros(()), {"note": "batch too small"}
    #
    #     # 1) 相似度（CLIP/SimCLR），tau 用 self.tau
    #     q = F.normalize(q, dim=1, eps=1e-8)
    #     k = F.normalize(k, dim=1, eps=1e-8)
    #     s = (q @ k.T) / self.tau  # [B,B]
    #     eye = torch.eye(B, device=s.device, dtype=torch.bool)
    #
    #     # 行/列 log-softmax 概率
    #     logp_row = s - torch.logsumexp(s, dim=1, keepdim=True)  # [B,B]
    #     logp_col = s.T - torch.logsumexp(s.T, dim=1, keepdim=True)  # [B,B]
    #
    #     # 2) 用“优势差”构造配对级软标签：|A_i - A_j|
    #     with torch.no_grad():
    #         A = adv.detach().view(B)  # [B]
    #         pair = (A[:, None] - A[None, :]).abs()  # [B,B], 对角=0
    #         off = pair[~eye]
    #         p95 = torch.quantile(off, 0.95) if off.numel() > 0 else torch.tensor(1.0, device=pair.device)
    #         pair_n = pair / p95.clamp_min(1e-6)  # 稳尺度，防极端值吃满 softmax
    #
    #     # 只在负样上归一：对角排除（-inf）
    #     w_row = torch.softmax(beta * pair_n.masked_fill(eye, float('-inf')), dim=1)  # i→t，每行和=1
    #     w_col = torch.softmax(beta * pair_n.masked_fill(eye, float('-inf')), dim=0)  # t→i，每列和=1
    #
    #     # 3) 软标签 CE（损失外）——对称
    #     loss_i2t = -(w_row * logp_row).sum(dim=1).mean()
    #     loss_t2i = -(w_col * logp_col).sum(dim=1).mean()
    #     loss = 0.5 * (loss_i2t + loss_t2i)
    #
    #     with torch.no_grad():
    #         stats = {
    #             "top1_i2t": (s.argmax(1) == torch.arange(B, device=s.device)).float().mean().item(),
    #             "top1_t2i": (s.T.argmax(1) == torch.arange(B, device=s.device)).float().mean().item(),
    #             "w_pair_p95": torch.quantile(w_row[~eye], 0.95).item() if (~eye).any() else 0.0,
    #         }
    #     return loss, stats

    def get_action_and_value(self, x, action=None, x_next=None, done=None, private_obs=None, org_x = None, alpha = 0.0, adv=None):

        org_state = x["state"].clone()
        x, feature_dict = self.feature_net(x)

        self.last_attn_maps = {}
        self.last_attn_logits = {}
        if hasattr(self.feature_net, "last_attn_maps"):
            m = self.feature_net.last_attn_maps
            if "rgb_0" in m: self.last_attn_maps["rgb_0"] = m["rgb_0"]
        if hasattr(self.feature_net, "last_attn_logits"):
            ml = self.feature_net.last_attn_logits
            if "rgb_0" in ml: self.last_attn_logits["rgb_0"] = ml["rgb_0"]  # 新增

        # 单目相机：只用rgb_0
        image_representation = feature_dict["rgb_0"]
        
        # recon_rep = self.image_fc(image_representation)
        z_img = self.img_proj(image_representation)
        z_state = self.state_proj(private_obs)


        if self.tcl_choice == 1:
            tcl, plot_info = self.info_nce_loss(z_img, z_state)
        elif self.tcl_choice == 2:
            tcl, plot_info = self.barlow_twins_loss(z_img, z_state)
        elif self.tcl_choice == 4:
            tcl, plot_info = self.adv_info_nce_loss(z_img, z_state, adv, use_logits=False, sigma=self.sigma)
        elif self.tcl_choice == 5:
            tcl, plot_info = self.adv_info_nce_loss(z_img, z_state, adv, use_logits=True, sigma=self.sigma)
        elif self.tcl_choice == 51:
            tcl, plot_info = self.adv_info_nce_loss(z_img, z_state, adv, use_logits=False, sigma=self.sigma,  label_smoothing=0.0)
        elif self.tcl_choice == 52:
            tcl, plot_info = self.adv_info_nce_loss(z_img, z_state, adv, use_logits=False, sigma=self.sigma,  label_smoothing=0.1)
        elif self.tcl_choice == 41:
            tcl, plot_info = self.adv_info_nce_loss(z_img, z_state, adv, use_logits=False, sigma=self.sigma, new_tau=0.05)
        elif self.tcl_choice == 42:
            tcl, plot_info = self.adv_info_nce_loss(z_img, z_state, adv, use_logits=False, sigma=self.sigma, new_tau=0.1)
        elif self.tcl_choice == 43:
            tcl, plot_info = self.adv_info_nce_loss(z_img, z_state, adv, use_logits=False, sigma=self.sigma, new_tau=0.01)
        elif self.tcl_choice == 44:
            tcl, plot_info = self.adv_info_nce_loss(z_img, z_state, adv, use_logits=False, sigma=self.sigma, label_smoothing=0.1)
        elif self.tcl_choice == 6:
            if adv is not None:
                adv = (adv - adv.min()) / (adv.max() - adv.min())
            tcl, plot_info = self.pure_adv_info_nce_loss(z_img, z_state, adv, adv_tau=self.sigma)
        elif self.tcl_choice == 7:
            tcl, plot_info = self.pure_adv_info_nce_loss(z_img, z_state, adv, adv_tau=self.sigma, method=self.method)

        else:
            assert self.tcl_choice == 3
            tcl, plot_info = self.siglip_bce_loss(z_img, z_state)
        
        
        if self.tcl_choice != 5:
            assert 1==2

        private_state = self.private_state_fc(z_img)
        x = torch.cat([feature_dict["state"], private_state], -1)

       # x = self.policy_representation(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        if self.critic_use_pi:
            z = self.critic_representation(torch.cat([org_state, private_obs], -1))
            value = self.critic(z)
        else:
            value = self.critic(x)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value, tcl, plot_info

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)
import imageio.v2 as imageio
def make_overlay(rgb_bhwc: torch.Tensor,
                 attn_b1hw: torch.Tensor = None,
                 gt_b1hw: torch.Tensor = None,
                 alpha: float = 0.6):
    """
    输入:
      rgb_bhwc: [B,H,W,3] uint8/float (0~255或0~1)
      attn_b1hw: [B,1,h',w'] (0~1)
      gt_b1hw:   [B,1,h',w'] (0/1或0~1)
    输出:
      (rgb_chw, pred_overlay, gt_overlay) 都是 [B,3,H,W], 0~1
    """
    B, H, W, _ = rgb_bhwc.shape
    rgb = rgb_bhwc.float()
    if rgb.max() > 1.1: rgb = rgb / 255.0
    rgb_chw = rgb.permute(0,3,1,2).contiguous()  # [B,3,H,W]

    pred_overlay = None
    gt_overlay = None

    if attn_b1hw is not None:
        A_up = F.interpolate(attn_b1hw, size=(H, W), mode='bilinear', align_corners=False).clamp(0,1)
        red = torch.zeros_like(rgb_chw); red[:,0:1,:,:] = 1.0
        pred_overlay = (1 - alpha*A_up) * rgb_chw + (alpha*A_up) * red
        pred_overlay = pred_overlay.clamp(0,1)

    if gt_b1hw is not None:
        M_up = F.interpolate(gt_b1hw, size=(H, W), mode='nearest').clamp(0,1)
        gt_overlay = rgb_chw.clone()
        gt_overlay[:,1:2,:,:] = torch.max(gt_overlay[:,1:2,:,:], 0.6*M_up)
        gt_overlay = gt_overlay.clamp(0,1)

    return rgb_chw, pred_overlay, gt_overlay


class MaskDebugDumper:
    """
    导出“原始 seg / 二值 GT / 下采样后重建”的对比条。默认只存图，不写视频。
    每步会输出 cam0_* / cam1_* 的 strip PNG（RGB | raw seg color | bin-GT overlay | recon overlay）
    """
    def __init__(self, outdir: str, use_video: bool=False, fps: int=20):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.use_video = use_video
        self.fps = fps
        self.writers = {}

    def _to_np(self, chw: torch.Tensor) -> np.ndarray:
        img = chw.clamp(0,1).permute(1,2,0).detach().cpu().numpy()
        return (img*255).astype(np.uint8)

    def _write_strip(self, tag: str, step: int, env_id: int, imgs_chw: list):
        strip = np.concatenate([self._to_np(x) for x in imgs_chw], axis=1)
        fn = os.path.join(self.outdir, f"{tag}_env{env_id:02d}_step{step:08d}.png")
        imageio.imwrite(fn, strip)
        if self.use_video:
            key = (tag, env_id)
            if key not in self.writers:
                self.writers[key] = imageio.get_writer(
                    os.path.join(self.outdir, f"{tag}_env{env_id:02d}.mp4"),
                    fps=self.fps, codec="libx264", quality=8)
            self.writers[key].append_data(strip)

    def write(self, step: int,
              rgb: torch.Tensor, raw_seg: torch.Tensor, down_b1hw: torch.Tensor,
              select_envs=[0], tag="cam0"):
        """
        rgb: [B,H,W,3] uint8/float
        raw_seg: [B,H,W] (long) 或 None
        down_b1hw: [B,1,h',w'] 或 None（就是你存进 attn_sup 的那个）
        """
        B,H,W,_ = rgb.shape
        rgb_chw = rgb.float()
        if rgb_chw.max() > 1.1: rgb_chw = rgb_chw/255.0
        rgb_chw = rgb_chw.permute(0,3,1,2).contiguous()  # [B,3,H,W]

        if raw_seg is not None:
            raw_color = _colorize_labels(raw_seg)                  # [B,3,H,W]
            gt_bin = _bin_from_raw_seg(raw_seg)                    # [B,1,H,W]
        else:
            raw_color = rgb_chw
            gt_bin = torch.zeros((B,1,H,W), device=rgb.device)

        # 重建：把下采样 mask 拉回到 HxW
        if down_b1hw is not None:
            up = F.interpolate(down_b1hw, size=(H,W), mode="nearest")
            recon_bin = (up > 0.0).float()
            recon_bin_1 = (up > 0.2).float()
            recon_bin_2 = (up > 0.5).float()
        else:
            recon_bin = torch.zeros_like(gt_bin)

        # 叠加：原始 GT 用绿色，重建用蓝色
        gt_overlay    = _overlay_mask(rgb_chw, gt_bin,    color_idx=1, alpha=0.6)  # G

        recon_overlay_1 = _overlay_mask(rgb_chw, recon_bin, color_idx=2, alpha=0.6)  # B
        recon_overlay_2 = _overlay_mask(rgb_chw, recon_bin_1, color_idx=2, alpha=0.6)  # B
        recon_overlay_3 = _overlay_mask(rgb_chw, recon_bin_2, color_idx=2, alpha=0.6)  # B

        for e in select_envs:
            imgs = [rgb_chw[e], recon_overlay_1[e], recon_overlay_2[e], recon_overlay_3[e]]
            self._write_strip(tag, step, e, imgs)

    def close(self):
        for w in self.writers.values():
            try: w.close()
            except: pass
        self.writers.clear()

def get_infos(env_id, infos):


    if env_id == "SO101GraspCube_two_cameras-v1": # Done
        private_info = torch.cat(
            [infos["obj_pose"], infos["tcp_pose"], infos["tcp_to_obj_pos"]], -1)
    elif env_id == "SO101GraspMovingCube_two_cameras-v1": # Done
        private_info = torch.cat(
            [infos["obj_pose"], infos["tcp_pose"], infos["tcp_to_obj_pos"]], -1)
    elif env_id == "SO101GraspYCB_two_cameras-v1": # Done
        private_info = torch.cat(
            [infos["obj_pose"], infos["tcp_pose"], infos["obj_pos"], infos["tcp_to_obj_pos"]], -1)
    elif env_id == "SO101GraspYCB_return_two_cameras-v1": # Done
        private_info = torch.cat(
            [infos["obj_pose"], infos["tcp_pose"], infos["obj_pos"], infos["tcp_to_obj_pos"]], -1)
    elif  env_id == "SO101GraspVisdex_return_two_cameras-v1": # Done
        private_info = torch.cat(
            [infos["obj_pose"], infos["tcp_pose"], infos["obj_pos"], infos["tcp_to_obj_pos"]], -1)
    elif env_id == "SO101LiftPegUprightEnvWithTwoCameras-v1": # Done

        private_info = torch.cat(
            [infos["obj_pose"], infos["tcp_pose"], infos["tcp_to_obj_pos"], infos["stand_distance"], infos["rot_distance"], infos["is_grasped"].unsqueeze(-1)], -1)
    elif env_id == "SO101PegInsertSide_two_cameras-v1": # Done
        private_info = torch.cat([infos["peg_pose"], infos["peg_half_size"], infos["tcp_pose"],infos["box_hole_pose"] ,infos["box_hole_radius"]], -1)
    elif env_id == "SO101PushT_two_cameras-v1": # Done
        private_info = torch.cat(
            [infos["tcp_pose"], infos["tee_pose"], infos["goal_tee_pose"], infos["inter_area"]], -1)
    elif env_id == "SO101PlaceSphere_two_cameras-v1": # Done
        private_info = torch.cat(
            [infos["is_obj_grasped"].unsqueeze(-1), infos["obj_pose"], infos["bin_pose"], infos["tcp_pose"], infos["tcp_to_obj_pos"]], -1)
    elif env_id == "SO101StackCube_two_cameras-v1":
        private_info = torch.cat(
            [infos["is_cubeA_grasped"].unsqueeze(-1), infos["cubeA_pose"], infos["cubeB_pose"], infos["tcp_pose"], infos["tcp_to_cubeA_pos"],
             infos["tcp_to_cubeB_pos"], infos["cubeA_to_cubeB_pos"]], -1)
    return private_info

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

def train(args: PPOArgs):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name + "_" + str(args.seed)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(
        obs_mode="rgb+segmentation", render_mode=args.render_mode, sim_backend="physx_cuda",
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    env_kwargs.update(args.env_kwargs)

    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    envs = Flatten_Multi_RGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = Flatten_Multi_RGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=eval_envs.unwrapped.control_freq)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=eval_envs.unwrapped.control_freq, info_on_video=True)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=args.exp_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ALGO Logic: Storage setup
    obs      = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    print(envs.single_observation_space)

    #org_obs =  DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)

    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    next_obs, infos = envs.reset(seed=args.seed)

    private_info = get_infos(args.env_id, infos)

    private_dim = private_info.shape[-1]

    private_obs =  torch.zeros((args.num_steps, args.num_envs, private_dim)).to(device)

    next_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    agent = Agent(args.method, args.sigma, envs, sample_obs=next_obs, net_type=args.net_type, choice=args.choice, tcl_choice=args.tcl_choice, device= device, z_detach = args.z_detach, private_dim= private_dim, search_entropy= args.search_entropy, act_type=args.act_type,  critic_use_pi=args.critic_use_pi).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # 在 agent/optimizer 创建之后
    attn_sup0 = None
    h_attn = w_attn = None
    alpha = 0.0
    current_success_rate = 0.0 
    if args.use_attn:
        with torch.no_grad():
            dummy_priv = torch.zeros((args.num_envs, private_dim), device=device)
            _a, _lp, _, _v, _t, _ = agent.get_action_and_value(next_obs, private_obs=dummy_priv, alpha = alpha )
        maps = getattr(agent, "last_attn_maps", {})
        Aref = maps.get("rgb_0", None)  # 单目相机：只用rgb_0
        if Aref is not None:
            h_attn, w_attn = Aref.shape[-2:]
            attn_sup0 = torch.zeros((args.num_steps, args.num_envs, 1, h_attn, w_attn), device=device)
        else:
            print("[Warn] use_attn=True 但未拿到注意力图；将跳过注意力监督。")

        fps_guess = 20
        try:
            fps_guess = int(eval_envs.unwrapped.control_freq)
        except Exception:
            pass
        outdir = f"runs/{run_name}/attn_rollout"

        mask_dumper = MaskDebugDumper(outdir=outdir,
                                      use_video=True,  # 图片足够，想要 mp4 改 True
                                      fps=fps_guess)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    cumulative_times = defaultdict(float)
    train_step_num = 0

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            stime = time.perf_counter()
            eval_obs, eval_infos = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    private_info = get_infos(args.env_id, eval_infos)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs,  private_state=private_info , deterministic=True, alpha = alpha))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            eval_success_rate = None
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)

                
                print(f"eval_{k}_mean={mean}")
                # 尝试获取成功率（常见的指标名称）
                if eval_success_rate is None:
                    if k in ["success_at_end"]:
                        eval_success_rate = mean.item()
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            # 在文件名中包含成功率（如果可用）
            if eval_success_rate is not None:
                success_str = f"_success{eval_success_rate:.3f}"
            else:
                success_str = ""
            model_path = f"runs/{run_name}/ckpt_{iteration}{success_str}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        rollout_time = time.perf_counter()



        if iteration < args.alpha* args.num_iterations :
            alpha = 0.0

        elif iteration > (1-args.end_alpha)* args.num_iterations:
            alpha = 1.0
        else:
            alpha = (iteration - args.alpha* args.num_iterations)/ ( (1- args.alpha - args.end_alpha)* args.num_iterations)
        
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
        #    org_obs[step] = next_obs

            if attn_sup0 is not None:
                m0 = masks_from_infos(infos, 0)  # [B,1,H,W] or None
                # 单目相机：只用rgb_0的mask
                if m0 is not None: attn_sup0[step] = downsample_mask(m0, h_attn, w_attn)
                else: assert 1==2

        #    print(step, "!!!!",infos.keys() )
        #    org_obs[step]["rgb_0"] = infos["org_rgb_0"]

            dones[step] = next_done

            private_obs[step] = private_info = get_infos(args.env_id, infos)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, _, _ = agent.get_action_and_value(next_obs, private_obs = private_obs[step], alpha = alpha)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)


#            if mask_dumper is not None and (step % 1 == 0):
#                raw0 = _find_raw_seg_from_obs(infos, 0)  # [B,H,W] or None
#                raw1 = _find_raw_seg_from_obs(infos, 1)
#
#                # 当前帧已经存进来的 “下采样到注意力分辨率的 GT”
#                down0 = attn_sup0[step] if attn_sup0 is not None else None  # [B,1,h',w'] or None
#                down1 = attn_sup1[step] if attn_sup1 is not None else None
#
#                sel_envs = [0]  # 只存第 0 个并行环境
#                mask_dumper.write(step=global_step, rgb=next_obs["rgb_0"], raw_seg=raw0, down_b1hw=down0,
#                                  select_envs=sel_envs, tag="cam0")
#                if "rgb_1" in next_obs:
#                    mask_dumper.write(step=global_step, rgb=next_obs["rgb_1"], raw_seg=raw1, down_b1hw=down1,
#                                      select_envs=sel_envs, tag="cam1")
#            assert  1== 2
            #
            # print( infos.keys())
            # # obj_pose = self.cube.pose.raw_pose,
            # # tcp_pos = self.agent.tcp_pos,
            # # obj_pos = self.cube.pose.p,
            # # tcp_to_obj_pos = self.cube.pose.p - self.agent.tcp_pos,
            # print("1", infos["obj_pose"].shape)
            # print("2", infos["tcp_pos"].shape)
            # print("3", infos["obj_pos"].shape)
            # print("4", infos["tcp_to_obj_pos"].shape)
            #assert  1== 2
            # for key in infos.keys():
            #     print(key, "_", infos[key].shape)
            #
            # print(infos.keys())

            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            next_private_state = get_infos(args.env_id, infos)

            next_dones[step] = next_done
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                
                    if k == "success_once":
                        current_success_rate = v[done_mask].float().mean()
                
                
                for k in infos["final_observation"]:
                    #print(k, "1", infos["final_observation"][k].shape, done_mask.shape, infos["final_private_state"].shape)
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                
                infos["final_private_state"] = infos["final_private_state"][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"], infos["final_private_state"]).view(-1)
                    
        # mask_dumper.close()

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_private_state).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))



        #b_org_obs = org_obs.reshape((-1,))
        b_next_done = next_dones.reshape(-1)
        b_private_obs = private_obs.reshape((-1, private_dim))

        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        
        if attn_sup0 is not None :
            b_attn_sup0 = attn_sup0.reshape(args.batch_size, 1, h_attn, w_attn)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)


        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)


                _, newlogprob, entropy, newvalue, tcl_loss, plot_info  = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], None, b_next_done[mb_inds], b_private_obs[mb_inds], None, alpha = alpha, adv = mb_advantages)


                # Add camera mask - 单目相机：只用rgb_0
                L_mask = torch.tensor(0.0, device=device);
                if attn_sup0 is not None :
                    last_attn_logits = getattr(agent, "last_attn_logits", {})
                    logits0 = last_attn_logits.get("rgb_0", None)

                    def _bce_logits(logits, target):
                        # target 是 0/1；logits 是未过sigmoid
                        # 计算一个动态 pos_weight（负例/正例），抑制"全0"解
                        with torch.no_grad():
                            p = target.mean()  # 正例比例
                            eps = 1e-6
                            pos_w = (1 - p + eps) / (p + eps)  # ~ N_neg / N_pos
                        return F.binary_cross_entropy_with_logits(
                            logits, target, pos_weight=pos_w)

                    if logits0 is not None:
                        b_sup0 = b_attn_sup0[mb_inds]
                        L_mask = _bce_logits(logits0, b_sup0)




                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = L_mask + args.tcl_weight * tcl_loss + pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                train_step_num +=1




            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        for key in plot_info.keys():
            logger.add_scalar("losses/" + str(key), plot_info[key].item(), global_step)

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/alpha", alpha, global_step)
        logger.add_scalar("losses/tcl_loss", tcl_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None: logger.close()