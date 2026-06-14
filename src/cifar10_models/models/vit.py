"""Vision Transformer (ViT) for CIFAR-10 with optional CNN stem and stochastic depth.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Patchify the input image with a strided convolution."""

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 256,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class CNNStem(nn.Module):
    """Small CNN stem to produce patch embeddings for ViT."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        image_size: int = 32,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size // 2, stride=patch_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        stochastic_depth: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth drop path."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob) / keep_prob
        return x * mask


class VisionTransformer(nn.Module):
    """Vision Transformer classifier for CIFAR-10."""

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        patch_size: int = 4,
        embed_dim: int = 256,
        num_heads: int = 4,
        depth: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        stochastic_depth: float = 0.0,
        use_cnn_stem: bool = False,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        channels, image_size, _ = image_shape

        if use_cnn_stem:
            self.patch_embed = CNNStem(
                in_channels=channels,
                embed_dim=embed_dim,
                image_size=image_size,
                patch_size=patch_size,
            )
        else:
            self.patch_embed = PatchEmbedding(
                in_channels=channels,
                patch_size=patch_size,
                embed_dim=embed_dim,
                image_size=image_size,
            )

        num_patches = self.patch_embed.num_patches
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

        self.dropout = nn.Dropout(dropout)
        sd_rates = [stochastic_depth * i / max(depth - 1, 1) for i in range(depth)]
        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    stochastic_depth=rate,
                )
                for rate in sd_rates
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.patch_embed(x)
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
