"""Geometry-Supervised Dual Routing (GSDR) modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv


# Scale statistics computed from the VisDrone training labels (381,963 boxes).
GSDR_SCALE_MEAN = -3.808831
GSDR_SCALE_STD = 0.750130
# Branch centers operate on density after calibration to the unit routing interval.
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
    """Foreground-gated density-scale routing for P2 while passing P3/P4 through unchanged."""

    def __init__(
        self,
        channels: Sequence[int],
        hidden_channels: int = 64,
        routing_temperature: float = 0.15,
        alpha_max: float = 0.15,
        warmup_epochs: float = 10.0,
        density_temperature: float = 2.0,
        density_route_floor: float = 0.05,
        density_route_ceiling: float = 0.30,
        calibrate_residual_gate: bool = False,
        detach_density_for_routing: bool = False,
        use_scale_for_routing: bool = False,
        uniform_routing: bool = False,
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("GSDR expects exactly three feature levels: P2, P3, and P4.")
        if min(channels) < 1 or hidden_channels < 8:
            raise ValueError("GSDR channel counts must be positive and hidden_channels must be >= 8.")
        if routing_temperature <= 0 or alpha_max < 0 or warmup_epochs < 0 or density_temperature <= 0:
            raise ValueError("GSDR routing, residual, warmup, or density settings are invalid.")
        if not 0 <= density_route_floor < density_route_ceiling <= 1:
            raise ValueError("GSDR density routing bounds must satisfy 0 <= floor < ceiling <= 1.")

        self.channels = tuple(int(c) for c in channels)
        self.hidden_channels = int(hidden_channels)
        self.routing_temperature = float(routing_temperature)
        self.alpha_max = float(alpha_max)
        self.warmup_epochs = float(warmup_epochs)
        self.density_temperature = float(density_temperature)
        self.density_route_floor = float(density_route_floor)
        self.density_route_ceiling = float(density_route_ceiling)
        self.calibrate_residual_gate = bool(calibrate_residual_gate)
        self.detach_density_for_routing = bool(detach_density_for_routing)
        self.use_scale_for_routing = bool(use_scale_for_routing)
        self.uniform_routing = bool(uniform_routing)
        self.current_epoch = 0.0

        self.p2_in_proj = Conv(self.channels[0], hidden_channels, 1, 1)
        self.p2_context_branches = nn.ModuleList(
            nn.Sequential(
                DWConv(hidden_channels, hidden_channels, 3, 1, d=dilation),
                Conv(hidden_channels, hidden_channels, 1, 1),
            )
            for dilation in (1, 2, 3)
        )
        self.p2_out_proj = Conv(hidden_channels, self.channels[0], 1, 1, act=False)

        prior_channels = max(16, min(hidden_channels, 32))
        self.prior_stem = Conv(self.channels[0], prior_channels, 3, 1)
        self.prior_head = nn.Conv2d(prior_channels, 2, 1)
        nn.init.constant_(self.prior_head.bias[0], -3.0)
        nn.init.constant_(self.prior_head.bias[1], 0.0)

        density_gaps = torch.tensor(
            [
                GSDR_DENSITY_CENTERS[0],
                GSDR_DENSITY_CENTERS[1] - GSDR_DENSITY_CENTERS[0],
                GSDR_DENSITY_CENTERS[2] - GSDR_DENSITY_CENTERS[1],
                1.0 - GSDR_DENSITY_CENTERS[2],
            ],
            dtype=torch.float32,
        )
        self.density_gap_logits = nn.Parameter(density_gaps.log())
        # Keep the residual small at initialization while leaving enough gradient for the routed branch to learn.
        self.residual_logits = nn.Parameter(torch.tensor([-1.5], dtype=torch.float32))
        self.register_buffer("target_kernel", make_gsdr_gaussian_kernel(), persistent=False)
        diagnostic_buffers = {
            "_diag_route_soft_sum": torch.zeros(3),
            "_diag_route_hard_count": torch.zeros(3),
            "_diag_positive_hard_count": torch.zeros(3),
            "_diag_negative_hard_count": torch.zeros(3),
            "_diag_gate_sum": torch.zeros(1),
            "_diag_pixel_count": torch.zeros(1),
            "_diag_positive_count": torch.zeros(1),
            "_diag_negative_count": torch.zeros(1),
            "_diag_delta_square_sum": torch.zeros(1),
            "_diag_p2_square_sum": torch.zeros(1),
        }
        self._diagnostic_buffer_names = tuple(diagnostic_buffers)
        for name, value in diagnostic_buffers.items():
            self.register_buffer(name, value, persistent=False)
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

    def consume_aux(self) -> dict[str, torch.Tensor] | None:
        """Return the latest training-only auxiliary maps and release their retained computation graph."""
        aux, self.last_aux = self.last_aux, None
        return aux

    def _route(self, value: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        distances = (value - centers.view(1, -1, 1, 1)).square()
        return torch.softmax(-distances / (2 * self.routing_temperature**2), dim=1)

    def calibrate_density(self, density: torch.Tensor) -> torch.Tensor:
        """Map supervised density values onto the full routing interval."""
        floor = getattr(self, "density_route_floor", None)
        ceiling = getattr(self, "density_route_ceiling", None)
        if floor is None or ceiling is None:
            return density  # Legacy v2 checkpoints keep their original uncalibrated routing behavior.
        return ((density - floor) / (ceiling - floor)).clamp(0, 1)

    def routing_signal(self, density: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
        """Return the calibrated geometry signal used to select context branches."""
        routing_density = self.calibrate_density(density)
        if not getattr(self, "use_scale_for_routing", False):
            return routing_density
        if scale is None:
            raise ValueError("Scale-aware GSDR routing requires a scale map.")
        if scale.shape != density.shape:
            raise ValueError("GSDR density and scale maps must have identical shapes for scale-aware routing.")

        # Density confidence controls how strongly inverse object scale shifts foreground routing. Background stays at
        # zero, while small objects move toward dilation 1 and large objects move toward dilations 2-3.
        inverse_scale = 1.0 - scale.clamp(0, 1)
        return (routing_density + routing_density * inverse_scale) / (1.0 + routing_density)

    def routing_weights(self, density: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
        """Return branch weights from the configured density or density-scale routing signal."""
        if getattr(self, "uniform_routing", False):
            # Keep all three context branches active while removing spatially adaptive branch selection.
            return density.new_full((density.shape[0], 3, *density.shape[-2:]), 1.0 / 3.0)
        routing_signal = self.routing_signal(density, scale)
        route_centers = _ordered_centers(self.density_gap_logits).flip(0).to(dtype=density.dtype)
        return self._route(routing_signal, route_centers)

    def routing_input(self, density: torch.Tensor) -> torch.Tensor:
        """Detach v5 routing density while leaving legacy checkpoint gradients unchanged."""
        if getattr(self, "detach_density_for_routing", False):
            return density.detach()
        return density

    def residual_gate(self, density: torch.Tensor) -> torch.Tensor:
        """Return raw-density gating for v4+ while preserving legacy v3 checkpoint behavior."""
        if getattr(self, "calibrate_residual_gate", True):
            return self.calibrate_density(density)
        return density

    @torch.no_grad()
    def _record_forward_diagnostics(
        self,
        route_weights: torch.Tensor,
        residual_gate: torch.Tensor,
        p2: torch.Tensor,
        routed_p2: torch.Tensor,
    ) -> None:
        """Accumulate device-side routing statistics without synchronizing each batch."""
        route_weights = route_weights.detach().float()
        hard_route = route_weights.argmax(dim=1)
        hard_count = torch.stack([(hard_route == index).sum() for index in range(3)]).to(dtype=torch.float32)
        delta = routed_p2.detach().float() - p2.detach().float()

        self._diag_route_soft_sum.add_(route_weights.sum(dim=(0, 2, 3)))
        self._diag_route_hard_count.add_(hard_count)
        self._diag_gate_sum.add_(residual_gate.detach().float().sum())
        self._diag_pixel_count.add_(residual_gate.numel())
        self._diag_delta_square_sum.add_(delta.square().sum())
        self._diag_p2_square_sum.add_(p2.detach().float().square().sum())

    @torch.no_grad()
    def record_target_diagnostics(
        self,
        route_weights: torch.Tensor,
        density_target: torch.Tensor,
        positive_threshold: float,
    ) -> None:
        """Accumulate target-conditioned branch usage for the current training epoch."""
        if route_weights.shape[0] != density_target.shape[0] or route_weights.shape[-2:] != density_target.shape[-2:]:
            raise ValueError("GSDR route weights and density targets must share batch and spatial dimensions.")
        hard_route = route_weights.detach().argmax(dim=1)
        positive = density_target.detach().squeeze(1) >= positive_threshold
        negative = ~positive
        positive_count = torch.stack([((hard_route == index) & positive).sum() for index in range(3)]).float()
        negative_count = torch.stack([((hard_route == index) & negative).sum() for index in range(3)]).float()

        self._diag_positive_hard_count.add_(positive_count)
        self._diag_negative_hard_count.add_(negative_count)
        self._diag_positive_count.add_(positive.sum())
        self._diag_negative_count.add_(negative.sum())

    @torch.no_grad()
    def reset_diagnostics(self) -> None:
        """Clear accumulated routing statistics before a new epoch."""
        for name in self._diagnostic_buffer_names:
            getattr(self, name).zero_()

    @torch.no_grad()
    def consume_diagnostics(self) -> dict[str, float]:
        """Return epoch routing statistics as host scalars and reset their accumulators."""
        pixel_count = float(self._diag_pixel_count.item())
        if pixel_count == 0:
            return {}

        positive_count = float(self._diag_positive_count.item())
        negative_count = float(self._diag_negative_count.item())
        route_soft = (self._diag_route_soft_sum / pixel_count).cpu().tolist()
        route_hard = (self._diag_route_hard_count / pixel_count).cpu().tolist()
        positive_hard = (self._diag_positive_hard_count / max(positive_count, 1.0)).cpu().tolist()
        negative_hard = (self._diag_negative_hard_count / max(negative_count, 1.0)).cpu().tolist()
        centers = _ordered_centers(self.density_gap_logits.detach()).flip(0).cpu().tolist()
        p2_delta_rms_ratio = torch.sqrt(
            self._diag_delta_square_sum / self._diag_p2_square_sum.clamp_min(1e-12)
        ).item()
        diagnostics = {
            "residual_gate_mean": float((self._diag_gate_sum / pixel_count).item()),
            "p2_delta_rms_ratio": float(p2_delta_rms_ratio),
            "positive_fraction": positive_count / max(positive_count + negative_count, 1.0),
            "effective_alpha": float(self.alpha_max * torch.sigmoid(self.residual_logits[0]) * self.warmup_factor()),
        }
        for index in range(3):
            dilation = index + 1
            diagnostics[f"route_soft_d{dilation}"] = float(route_soft[index])
            diagnostics[f"route_hard_d{dilation}"] = float(route_hard[index])
            diagnostics[f"positive_hard_d{dilation}"] = float(positive_hard[index])
            diagnostics[f"negative_hard_d{dilation}"] = float(negative_hard[index])
            diagnostics[f"route_center_d{dilation}"] = float(centers[index])
        self.reset_diagnostics()
        return diagnostics

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Return routed P2/P3/P4 features while retaining auxiliary prior predictions."""
        if not isinstance(features, (list, tuple)) or len(features) != 3:
            raise ValueError("GSDR expects a list containing P2, P3, and P4 feature maps.")
        if any(feature.ndim != 4 for feature in features):
            raise ValueError("Each GSDR feature map must have shape [B, C, H, W].")

        p2, p3, p4 = features
        p2_projected = self.p2_in_proj(p2)
        prior = self.prior_head(self.prior_stem(p2))
        density_map = prior[:, 0:1].sigmoid()
        scale_map = prior[:, 1:2].sigmoid()
        route_weights = self.routing_weights(self.routing_input(density_map), scale_map)
        residual_gate = self.residual_gate(density_map)
        # Model construction and inference run this module in eval mode before ModelEMA deep-copies the network.
        # Keep autograd-connected auxiliary maps only for the immediately following training loss calculation.
        self.last_aux = (
            {"density": density_map, "scale": scale_map, "route_weights": route_weights} if self.training else None
        )
        weights = route_weights.split(1, dim=1)
        p2_context = weights[0] * self.p2_context_branches[0](p2_projected)
        for index, branch in enumerate(self.p2_context_branches[1:], start=1):
            p2_context = p2_context + weights[index] * branch(p2_projected)

        alpha = self.alpha_max * torch.sigmoid(self.residual_logits[0]) * self.warmup_factor()
        routed_p2 = p2 + alpha * residual_gate * self.p2_out_proj(p2_context)
        if self.training:
            self._record_forward_diagnostics(route_weights, residual_gate, p2, routed_p2)
        return [routed_p2, p3, p4]
