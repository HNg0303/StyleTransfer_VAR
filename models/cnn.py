"""Pretrained CNN features for content and style representations."""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


__all__ = ['Normalization', 'VGG16Encoder', 'gram_matrix']


class Normalization(nn.Module):
    """Normalize floating-point RGB images in [0, 1] with (image - mean) / std.

    Mean and standard deviation are fixed per-channel ImageNet statistics, not
    statistics calculated from the current image. Buffers follow ``.to(device)``
    and are included in the module's state dict.
    """

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).detach().clone()
        std = torch.as_tensor(std, dtype=torch.float32).detach().clone()
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError('mean and std must each contain three RGB values.')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError('mean and std must contain only finite values.')
        if (std <= 0).any():
            raise ValueError('std must be strictly positive for every channel.')

        self.register_buffer('mean', mean.view(1, 3, 1, 1))
        self.register_buffer('std', std.view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Expected RGB images with shape (B, 3, H, W).')
        if not image.is_floating_point():
            raise TypeError('Expected floating-point images, not integer pixels.')
        # Preserve the input dtype; device placement is handled by module.to().
        return (image - self.mean.to(dtype=image.dtype)) / self.std.to(dtype=image.dtype)


class VGG16Encoder(nn.Module):
    """Extract spatial VGG16 features after ImageNet normalization.

    Args:
        weights: Pretrained ImageNet weights by default. ``None`` creates random
            weights for offline tests; it is not useful for perceptual features.
        input_range: ``(0, 1)`` for RGB image tensors, or ``(-1, 1)`` for VAR/VAE
            images. The latter are converted to [0, 1] before normalization.
            Do not pass images that have already been ImageNet-normalized.
        freeze: Freeze encoder parameters and start in eval mode. Gradients with
            respect to the input image remain available for perceptual losses.
        progress: Display the pretrained-weight download progress, if needed.

    Returns:
        A dict of ``relu1_2``, ``relu2_2``, ``relu3_3``, ``relu4_3``, and
        ``relu5_3`` feature maps, with channels 64, 128, 256, 512, and 512.
        Height and width are reduced by factors 1, 2, 4, 8, and 16 (rounded down
        at each pooling step). Input height and width must be at least 16.

    The classifier and final max-pool are omitted. Images are not resized or
    center-cropped, preserving their spatial layout for style/content work.
    Attach this encoder after VAR.init_weights(), or exclude it from that
    initializer, which would otherwise overwrite its pretrained convolutions.
    """

    feature_layers = {
        3: 'relu1_2',
        8: 'relu2_2',
        15: 'relu3_3',
        22: 'relu4_3',
        29: 'relu5_3',
    }

    def __init__(
        self,
        weights: Optional[VGG16_Weights] = VGG16_Weights.IMAGENET1K_V1,
        input_range: Tuple[float, float] = (0, 1),
        freeze: bool = True,
        progress: bool = True,
    ):
        super().__init__()
        self.input_range = tuple(input_range)
        if self.input_range not in ((0, 1), (-1, 1)):
            raise ValueError('input_range must be (0, 1) or (-1, 1).')

        # Use the statistics paired with the checkpoint, including when callers
        # explicitly choose the alternate IMAGENET1K_FEATURES weights.
        preprocessing = (weights or VGG16_Weights.IMAGENET1K_V1).transforms()
        self.normalization = Normalization(preprocessing.mean, preprocessing.std)
        self.features = vgg16(weights=weights, progress=progress).features[:30]
        for layer in self.features:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False
        self.features.requires_grad_(not freeze)
        if freeze:
            self.eval()

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not image.is_floating_point():
            raise TypeError('Expected floating-point images, not integer pixels.')
        if self.input_range == (-1, 1):
            image = (image + 1) * 0.5
        x = self.normalization(image)
        if min(x.shape[-2:]) < 16:
            raise ValueError('VGG16Encoder requires height and width of at least 16.')

        features = {}
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.feature_layers:
                features[self.feature_layers[index]] = x
        return features


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix of a feature map.

    Args:
        features: A batch of feature maps with shape (B, C, H, W).

    Returns:
        The Gram matrix for each feature map, with shape (B, C, C).
        Half/bfloat16 inputs accumulate in float32, even under autocast.
    """
    if features.ndim != 4:
        raise ValueError('Expected 4D feature maps (B, C, H, W).')
    B, C, H, W = features.shape
    with torch.autocast(device_type=features.device.type, enabled=False):
        if features.dtype in (torch.float16, torch.bfloat16):
            features = features.float()
        features = features.reshape(B, C, H * W)
        # Each image has its own statistics; never correlate different batch items.
        return features.bmm(features.transpose(1, 2)).div(C * H * W)


if __name__ == '__main__':
    # Quick test of the VGG16Encoder and Gram matrix computation.
    encoder = VGG16Encoder(weights=VGG16_Weights.IMAGENET1K_V1, freeze=False)
    test_image = torch.rand(1, 3, 64, 64)  # Random image in [0, 1]
    features = encoder(test_image)
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
        if name == "relu5_3":  # Example: compute Gram matrix for the last feature map
            G = gram_matrix(feat)
            print(f"Gram matrix for {name}: {G.shape}")
