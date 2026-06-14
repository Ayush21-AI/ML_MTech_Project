"""Patch-based CNN for CIFAR-10.

Extracts non-overlapping patches and applies shared convolutional layers
independently to each patch before classifying the concatenated features.
"""

import torch
from torch import nn


class PatchBasedCNN(nn.Module):
    """Patch-based convolutional neural network.

    Parameters
    ----------
    image_shape: tuple[int, int, int]
        ``(channels, height, width)`` input shape. Default ``(3, 32, 32)``.
    patch_size: int
        Side length of each square patch. Default ``4``.
    embed_dim: int
        Number of channels after the initial patch projection. Default ``128``.
    depth: int
        Number of patch-processing blocks. Default ``6``.
    num_classes: int
        Number of output classes. Default ``10``.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 6,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        channels, height, width = image_shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("patch_size must divide both image height and width")

        self.patch_size = patch_size
        self.num_patches = (height // patch_size) * (width // patch_size)

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        # Shared patch CNN.
        self.patch_cnn = nn.Sequential(
            nn.Conv2d(channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, patch_size, patch_size)
            patch_feature_dim = self.patch_cnn(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(self.num_patches * patch_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        patches = self.unfold(x)  # (B, C*P*P, num_patches)
        patches = patches.permute(0, 2, 1)
        patches = patches.reshape(
            batch_size * self.num_patches,
            x.shape[1],
            self.patch_size,
            self.patch_size,
        )

        features = self.patch_cnn(patches)
        features = features.reshape(batch_size, self.num_patches * features.shape[1])
        return self.classifier(features)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
