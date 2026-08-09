"""Geometry-Supervised Dual Routing (GSDR) modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv


# Statistics computed from the VisDrone training labels (381,963 boxes).
GSDR_SCALE_MEAN = -3.808831
GSDR_SCALE_STD = 0.750130
GSDR_SCALE_CENTERS = (0.334, 0.493, 0.666)
GSDR_DENSITY_CENTERS = (0.25, 0.48, 0.67)


def make_gsdr_gaussian_kernel(size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2-D Gaussian kernel with a unit peak for target-map smoothing."""
    if size < 3 or size % 2 == 0:
        raise ValueError("GSDR Gaussian kernel size must be an odd integer >= 3.")
    if sigma <= 0:
        raise ValueError("GSDR Gaussian kernel sigma must be positive.")
    radius = size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma**2))
    return kernel.view(1, 1, size, size)


def build_gsdr_targets(
    batch_idx: torch.Tensor,
    boxes: torch.Tensor,
    batch_size: int,
    output_size: tuple[int, int],
    kernel: torch.Tensor | None = None,
    density_temperature: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build density, normalized scale, and scale-mask targets from normalized ``xywh`` boxes.

    The target maps are generated from the transformed boxes that reach the detector. A center impulse is smoothed by
    a fixed Gaussian kernel, so nearby objects naturally produce a stronger density response without a per-batch
    pairwise nearest-neighbor computation.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [N, 4] in normalized xywh format.")
    if batch_idx.ndim != 1 or batch_idx.shape[0] != boxes.shape[0]:
        raise ValueError("batch_idx must have shape [N] matching boxes.")
    if len(output_size) != 2 or min(output_size) < 1:
        raise ValueError("output_size must be a positive (height, width) pair.")
    if density_temperature <= 0:
        raise ValueError("density_temperature must be positive.")

    height, width = output_size
    device = boxes.device
    dtype = boxes.dtype if boxes.is_floating_point() else torch.float32
    boxes = boxes.to(dtype=dtype)
    batch_idx = batch_idx.to(device=device, dtype=torch.long)
    density_impulse = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
    scale_impulse = torch.zeros_like(density_impulse)
    scale_weight = torch.zeros_like(density_impulse)

    if boxes.numel() == 0:
        zeros = torch.zeros_like(density_impulse)
        return zeros, zeros.clone(), zeros.clone()

    centers = boxes[:, :2].clamp(0, 1)
    x_index = (centers[:, 0] * width).long().clamp_(0, width - 1)
    y_index = (centers[:, 1] * height).long().clamp_(0, height - 1)
    flat_index = batch_idx * (height * width) + y_index * width + x_index

    density_impulse.view(-1).scatter_add_(0, flat_index, torch.ones_like(x_index, dtype=dtype))
    equivalent_size = boxes[:, 2:4].clamp_min(1e-6).prod(dim=1).sqrt()
    scale_value = torch.sigmoid((equivalent_size.log() - GSDR_SCALE_MEAN) / GSDR_SCALE_STD)
    scale_impulse.view(-1).scatter_add_(0, flat_index, scale_value)
    scale_weight.view(-1).scatter_add_(0, flat_index, torch.ones_like(scale_value))

    if kernel is None:
        kernel = make_gsdr_gaussian_kernel().to(device=device, dtype=dtype)
    else:
        kernel = kernel.to(device=device, dtype=dtype)
    padding = kernel.shape[-1] // 2
    density_blurred = F.conv2d(density_impulse, kernel, padding=padding)
    scale_blurred = F.conv2d(scale_impulse, kernel, padding=padding)
    weight_blurred = F.conv2d(scale_weight, kernel, padding=padding)

    density = 1.0 - torch.exp(-density_blurred / density_temperature)
    scale = scale_blurred / weight_blurred.clamp_min(1e-6)
    scale_mask = weight_blurred.clamp(0, 1)
    return density.clamp(0, 1), scale.clamp(0, 1), scale_mask


def _ordered_centers(logits: torch.Tensor) -> torch.Tensor:
    """Turn four learnable gap logits into three ordered points in the unit interval."""
    gaps = torch.softmax(logits, dim=0)
    return torch.cumsum(gaps, dim=0)[:-1]


class GSDR(nn.Module):
    """Geometry-Supervised Dual Routing for final P2/P3/P4 detection features."""

    def __init__(
        self,
        channels: Sequence[int],
        hidden_channels: int = 64,
        routing_temperature: float = 0.15,
        alpha_max: float = 0.25,
        warmup_epochs: float = 3.0,
        density_temperature: float = 2.0,
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("GSDR expects exactly three feature levels: P2, P3, and P4.")
        if min(channels) < 1 or hidden_channels < 8:
            raise ValueError("GSDR channel counts must be positive and hidden_channels must be >= 8.")
        if routing_temperature <= 0 or alpha_max < 0 or warmup_epochs < 0 or density_temperature <= 0:
            raise ValueError("GSDR routing, residual, warmup, or density settings are invalid.")

        self.channels = tuple(int(c) for c in channels)
        self.hidden_channels = int(hidden_channels)
        self.routing_temperature = float(routing_temperature)
        self.alpha_max = float(alpha_max)
        self.warmup_epochs = float(warmup_epochs)
        self.density_temperature = float(density_temperature)
        self.current_epoch = 0.0

        self.in_proj = nn.ModuleList(Conv(c, hidden_channels, 1, 1) for c in self.channels)
        self.context_branches = nn.ModuleList(
            nn.ModuleList(
                nn.Sequential(
                    DWConv(hidden_channels, hidden_channels, 3, 1, d=dilation),
                    Conv(hidden_channels, hidden_channels, 1, 1),
                )
                for dilation in (1, 2, 3)
            )
            for _ in self.channels
        )
        self.out_proj = nn.ModuleList(
            Conv(hidden_channels, c, 1, 1, act=False) for c in self.channels
        )

        prior_channels = max(16, min(hidden_channels, 32))
        self.prior_stem = Conv(self.channels[0], prior_channels, 3, 1)
        self.prior_head = nn.Conv2d(prior_channels, 2, 1)
        nn.init.constant_(self.prior_head.bias[0], -3.0)
        nn.init.constant_(self.prior_head.bias[1], 0.0)

        scale_gaps = torch.tensor(
            [GSDR_SCALE_CENTERS[0], GSDR_SCALE_CENTERS[1] - GSDR_SCALE_CENTERS[0],
             GSDR_SCALE_CENTERS[2] - GSDR_SCALE_CENTERS[1], 1.0 - GSDR_SCALE_CENTERS[2]],
            dtype=torch.float32,
        )
        density_gaps = torch.tensor(
            [GSDR_DENSITY_CENTERS[0], GSDR_DENSITY_CENTERS[1] - GSDR_DENSITY_CENTERS[0],
             GSDR_DENSITY_CENTERS[2] - GSDR_DENSITY_CENTERS[1], 1.0 - GSDR_DENSITY_CENTERS[2]],
            dtype=torch.float32,
        )
        self.scale_gap_logits = nn.Parameter(scale_gaps.log())
        self.density_gap_logits = nn.Parameter(density_gaps.log())
        self.residual_logits = nn.Parameter(torch.full((3,), -4.0))
        self.register_buffer("target_kernel", make_gsdr_gaussian_kernel(), persistent=False)
        self.last_aux: dict[str, torch.Tensor] | None = None

    def set_epoch(self, epoch: int | float) -> None:
        """Set the training epoch used by the residual warmup."""
        self.current_epoch = float(epoch)

    def _warmup_factor(self) -> float:
        if not self.training or self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, (self.current_epoch + 1.0) / self.warmup_epochs)

    def warmup_factor(self) -> float:
        """Return the current residual and geometry-loss warmup factor."""
        return self._warmup_factor()

    def _route(self, value: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        distances = (value - centers.view(1, -1, 1, 1)).square()
        return torch.softmax(-distances / (2 * self.routing_temperature**2), dim=1)

    def _align(self, feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if feature.shape[-2:] == size:
            return feature
        mode = "nearest" if feature.shape[-2] < size[0] else "area"
        return F.interpolate(feature, size=size, mode=mode)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Return routed P2/P3/P4 features while retaining auxiliary prior predictions."""
        if not isinstance(features, (list, tuple)) or len(features) != 3:
            raise ValueError("GSDR expects a list containing P2, P3, and P4 feature maps.")
        if any(feature.ndim != 4 for feature in features):
            raise ValueError("Each GSDR feature map must have shape [B, C, H, W].")

        projected = [projection(feature) for projection, feature in zip(self.in_proj, features)]
        prior = self.prior_head(self.prior_stem(features[0]))
        density_map = prior[:, 0:1].sigmoid()
        scale_map = prior[:, 1:2].sigmoid()
        self.last_aux = {"density": density_map, "scale": scale_map}

        scale_centers = _ordered_centers(self.scale_gap_logits).to(dtype=scale_map.dtype)
        density_centers = _ordered_centers(self.density_gap_logits).flip(0).to(dtype=density_map.dtype)
        scale_weights = self._route(scale_map, scale_centers)

        context_features = []
        for level, (feature, branches) in enumerate(zip(projected, self.context_branches)):
            level_density = F.interpolate(density_map, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            density_weights = self._route(level_density, density_centers)
            context = sum(
                weight * branch(feature)
                for weight, branch in zip(density_weights.split(1, dim=1), branches)
            )
            context_features.append(context)

        routed = []
        for level, (original, output_projection) in enumerate(zip(features, self.out_proj)):
            target_size = projected[level].shape[-2:]
            neighbors = range(max(0, level - 1), min(3, level + 2))
            level_weights = F.interpolate(scale_weights, size=target_size, mode="bilinear", align_corners=False)
            selected_weights = torch.stack([level_weights[:, source] for source in neighbors], dim=1)
            selected_weights = selected_weights / selected_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            fused = sum(
                weight.unsqueeze(1) * self._align(context_features[source], target_size)
                for weight, source in zip(selected_weights.unbind(1), neighbors)
            )
            alpha = self.alpha_max * torch.sigmoid(self.residual_logits[level]) * self.warmup_factor()
            routed.append(original + alpha * output_projection(fused))
        return routed
