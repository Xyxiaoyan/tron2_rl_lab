"""Sensor encoder for RGB-D cameras + LiDAR observation encoding.

This module provides a SensorEncoder that takes bundled, downsampled
image and LiDAR observations and produces compact latent features.
The latents are then concatenated with proprioceptive observations
before being fed to the Actor/Critic MLPs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class SensorEncoder(nn.Module):
    """Encodes RGB-D images + LiDAR into compact feature vectors.

    图像输入: (N, 4*4*24*32) = (N, 3072) flat
     - 4 images (head-rgb, head-depth, down-rgb, down-depth), each 24×32
    LiDAR输入: (N, 24*72) = (N, 1728) flat
     - 降采样到 24×72

    输出: (N, image_latent_dim + lidar_latent_dim)
    """

    is_mlp_encoder = False
    is_sensor_encoder = True
    is_vae = False

    def __init__(
        self,
        image_h: int = 24,
        image_w: int = 32,
        image_c: int = 4,   # 4 images per bundle (head-rgb, head-depth, down-rgb, down-depth)
        image_ch: int = 4,  # channels per image (3 rgb + 1 depth)
        image_latent_dim: int = 64,
        lidar_rows: int = 24,
        lidar_cols: int = 72,
        lidar_latent_dim: int = 32,
        activation: str = "elu",
        orthogonal_init: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "SensorEncoder.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.image_latent_dim = image_latent_dim
        self.lidar_latent_dim = lidar_latent_dim
        self.total_latent_dim = image_latent_dim + lidar_latent_dim

        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c        # how many images
        self.image_ch = image_ch      # channels per image
        self.lidar_rows = lidar_rows
        self.lidar_cols = lidar_cols

        act = get_activation(activation)

        # ---- Image CNN ----
        # Reshape flat (N, C*ch*H*W) → (N, C*ch, H, W) then conv
        cnn_layers = []
        in_ch = image_c * image_ch  # 4*4=16 channels
        cnn_layers.append(nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1))
        cnn_layers.append(nn.BatchNorm2d(32))
        cnn_layers.append(act)
        cnn_layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        cnn_layers.append(nn.BatchNorm2d(64))
        cnn_layers.append(act)
        cnn_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1))
        cnn_layers.append(nn.BatchNorm2d(64))
        cnn_layers.append(act)
        cnn_layers.append(nn.AdaptiveAvgPool2d(1))
        self.cnn = nn.Sequential(*cnn_layers)
        # AdaptiveAvgPool2d(1) output: (N, 64, 1, 1) → flatten → 64
        cnn_flatten_dim = 64

        self.image_fc = nn.Sequential(
            nn.Linear(cnn_flatten_dim, image_latent_dim),
            act,
        )

        if orthogonal_init:
            self._orthogonal_init(self.image_fc[0], gain=1.0)

        # ---- LiDAR MLP ----
        lidar_flat_dim = lidar_rows * lidar_cols
        self.lidar_enc = nn.Sequential(
            nn.Linear(lidar_flat_dim, 128),
            act,
            nn.Linear(128, lidar_latent_dim),
            act,
        )

        if orthogonal_init:
            self._orthogonal_init(self.lidar_enc[0], gain=1.0)
            self._orthogonal_init(self.lidar_enc[2], gain=1.0)

        print(f"SensorEncoder CNN: {self.cnn}")
        print(f"SensorEncoder total_latent_dim: {self.total_latent_dim}")

    @staticmethod
    def _orthogonal_init(linear, gain=1.0):
        nn.init.orthogonal_(linear.weight, gain)
        nn.init.constant_(linear.bias, 0.0)

    def encode(self, image_bundle: torch.Tensor, lidar_bundle: torch.Tensor) -> torch.Tensor:
        """Encode bundled sensor data into latent features.

        Args:
            image_bundle: (N, C*ch*H*W) flat image tensor.
            lidar_bundle: (N, lidar_rows*lidar_cols) flat lidar tensor.

        Returns:
            (N, total_latent_dim) concatenated latent features.
        """
        n = image_bundle.shape[0]
        # Reshape image: (N, C*ch*H*W) → (N, C*ch, H, W)
        img = image_bundle.view(n, self.image_c * self.image_ch, self.image_h, self.image_w)
        cnn_out = self.cnn(img)          # (N, 64, 1, 1)
        cnn_out = cnn_out.flatten(1)     # (N, 64)
        image_latent = self.image_fc(cnn_out)  # (N, image_latent_dim)

        lidar_latent = self.lidar_enc(lidar_bundle)  # (N, lidar_latent_dim)

        return torch.cat([image_latent, lidar_latent], dim=-1)

    def forward(self, image_bundle: torch.Tensor, lidar_bundle: torch.Tensor) -> torch.Tensor:
        return self.encode(image_bundle, lidar_bundle)


class DummySensorEncoder(nn.Module):
    """No-op sensor encoder used when sensors are disabled.
    Returns a zero latent of dimension 0.
    """

    is_mlp_encoder = False
    is_sensor_encoder = True
    is_vae = False

    def __init__(self, **kwargs):
        super().__init__()
        self.total_latent_dim = 0

    def encode(self, *args, **kwargs):
        batch_size = args[0].shape[0]
        return torch.empty(batch_size, 0, device=args[0].device)

    def forward(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


def get_activation(act_name: str) -> nn.Module:
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CELU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print(f"Invalid activation function: {act_name}, using ReLU")
        return nn.ReLU()