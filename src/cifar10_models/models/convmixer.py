"""ConvMixer for CIFAR-10.

ConvMixer ("Patches Are All You Need?") keeps the patch-embedding stem of a
Vision Transformer but replaces the transformer blocks with depthwise +
pointwise convolutions. It is one of the strongest pure patch-based CNN
baselines for CIFAR-10.
"""

import torch
from torch import nn


class Residual(nn.Module):
    """Residual wrapper."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class ConvMixer(nn.Module):
    """ConvMixer classifier for CIFAR-10.

    Parameters
    ----------
    dim: int
        Channel width throughout the network. Default ``256``.
    depth: int
        Number of ConvMixer blocks. Default ``8``.
    kernel_size: int
        Kernel size of the depthwise convolution. Default ``5``.
    patch_size: int
        Patch embedding stride. Default ``2``.
    num_classes: int
        Number of output classes. Default ``10``.
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 8,
        kernel_size: int = 5,
        patch_size: int = 2,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                dim,
                                dim,
                                kernel_size=kernel_size,
                                groups=dim,
                                padding="same",
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(dim),
                        )
                    ),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )
                for _ in range(depth)
            ]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
