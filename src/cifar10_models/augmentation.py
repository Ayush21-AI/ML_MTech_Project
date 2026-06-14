"""Torchvision transforms and CutMix/Mixup helpers for CIFAR-10."""

from __future__ import annotations

import torch
from torchvision import transforms
from torchvision.transforms import v2

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_train_transforms(aug_cfg) -> transforms.Compose:
    """Build the training transform pipeline from an AugmentationConfig."""
    ops: list = []

    if aug_cfg.random_crop:
        ops.append(transforms.RandomCrop(32, padding=aug_cfg.random_crop_padding))

    if aug_cfg.random_horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip())

    if aug_cfg.randaugment:
        ops.append(
            transforms.RandAugment(
                num_ops=aug_cfg.randaugment_n,
                magnitude=aug_cfg.randaugment_m,
            )
        )

    if aug_cfg.color_jitter > 0:
        ops.append(
            transforms.ColorJitter(
                aug_cfg.color_jitter,
                aug_cfg.color_jitter,
                aug_cfg.color_jitter,
            )
        )

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD))

    if aug_cfg.random_erasing > 0:
        ops.append(transforms.RandomErasing(p=aug_cfg.random_erasing))

    return transforms.Compose(ops)


def build_test_transforms() -> transforms.Compose:
    """Build the validation/test transform pipeline."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
        ]
    )


class CutMixMixupCollator:
    """Batched CutMix/Mixup collate function for soft targets.

    Combines torchvision v2 CutMix and Mixup. Only one of the two is applied
    per sample, controlled by the underlying probabilities.
    """

    def __init__(
        self,
        cutmix_alpha: float = 1.0,
        mixup_alpha: float = 0.2,
        num_classes: int = 10,
        cutmix: bool = True,
        mixup: bool = True,
    ) -> None:
        self.num_classes = num_classes
        transforms_list: list = []
        if cutmix:
            transforms_list.append(
                v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes)
            )
        if mixup:
            transforms_list.append(
                v2.MixUp(alpha=mixup_alpha, num_classes=num_classes)
            )
        if transforms_list:
            self.transform = v2.RandomChoice(transforms_list)
        else:
            self.transform = None

    def __call__(self, batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

        if self.transform is None:
            return images, labels

        return self.transform(images, labels)
