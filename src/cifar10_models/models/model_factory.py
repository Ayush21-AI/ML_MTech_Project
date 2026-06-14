"""Factory for creating CIFAR-10 models by name."""

from cifar10_models.config import ModelConfig
from cifar10_models.models.convmixer import ConvMixer
from cifar10_models.models.patch_cnn import PatchBasedCNN
from cifar10_models.models.resnet import ResNet18
from cifar10_models.models.vit import VisionTransformer

_REGISTRY: dict[str, type] = {
    "patch_cnn": PatchBasedCNN,
    "convmixer": ConvMixer,
    "vit": VisionTransformer,
    "resnet18": ResNet18,
}


def list_models() -> list[str]:
    """Return the names of available models."""
    return list(_REGISTRY.keys())


def build_model(config: ModelConfig) -> nn.Module:
    """Build a model from a ModelConfig.

    Parameters
    ----------
    config: ModelConfig
        Model architecture configuration.

    Returns
    -------
    nn.Module
        The instantiated model with a ``count_parameters`` method.
    """
    if config.name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{config.name}'. Available: {list_models()}"
        )

    if config.name == "patch_cnn":
        return PatchBasedCNN(
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_classes=config.num_classes,
        )
    if config.name == "convmixer":
        return ConvMixer(
            dim=config.embed_dim,
            depth=config.depth,
            kernel_size=config.kernel_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
        )
    if config.name == "vit":
        return VisionTransformer(
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            depth=config.depth,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            stochastic_depth=config.stochastic_depth,
            use_cnn_stem=config.use_cnn_stem,
            num_classes=config.num_classes,
        )
    return ResNet18(num_classes=config.num_classes)
