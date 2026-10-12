from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

QUALITY_LABELS = (
    "blur",
    "glare",
    "shadow",
    "low_resolution",
    "partial_crop",
    "compression",
    "perspective",
)

MODEL_ARCHITECTURE = "mobilenet-v3-small-corner-heatmap-v1"


class DocumentPerceptionNet(nn.Module):
    """Shared visual encoder with independent document-evidence heads."""

    def __init__(
        self,
        *,
        num_document_types: int,
        pretrained: bool = True,
        hidden_features: int = 256,
    ) -> None:
        super().__init__()
        if num_document_types < 2:
            raise ValueError("at least two document classes are required")
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        feature_count = base.classifier[0].in_features
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_count, hidden_features),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
        )
        self.presence_head = nn.Linear(hidden_features, 1)
        self.corner_decoder = nn.Sequential(
            nn.Conv2d(feature_count, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.Hardswish(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Hardswish(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.Hardswish(),
            nn.Conv2d(32, 4, kernel_size=1),
        )
        self.corner_head = nn.Linear(hidden_features, 8)
        self.type_head = nn.Linear(hidden_features, num_document_types)
        self.quality_head = nn.Linear(hidden_features, len(QUALITY_LABELS))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.features(images)
        encoded = self.shared(self.pool(feature_map))
        corner_heatmaps = self.corner_decoder(feature_map)
        batch_size, corner_count, height, width = corner_heatmaps.shape
        probabilities = torch.softmax(
            corner_heatmaps.reshape(batch_size, corner_count, -1), dim=-1
        )
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=images.device, dtype=images.dtype),
            torch.linspace(0.0, 1.0, width, device=images.device, dtype=images.dtype),
            indexing="ij",
        )
        base_x = (probabilities * x_grid.flatten()).sum(dim=-1)
        base_y = (probabilities * y_grid.flatten()).sum(dim=-1)
        base_corners = torch.stack((base_x, base_y), dim=-1)
        residuals = 0.3 * torch.tanh(self.corner_head(encoded)).reshape(-1, 4, 2)
        corners = base_corners + residuals
        return {
            "presence_logits": self.presence_head(encoded).squeeze(-1),
            "corners": corners,
            "corner_heatmaps": corner_heatmaps,
            "type_logits": self.type_head(encoded),
            "quality_logits": self.quality_head(encoded),
            "embedding": encoded,
        }


def document_multitask_loss(
    output: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    corner_weight: float = 5.0,
    type_weight: float = 1.0,
    quality_weight: float = 0.5,
    heatmap_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    presence = target["presence"].to(dtype=output["presence_logits"].dtype)
    presence_loss = F.binary_cross_entropy_with_logits(
        output["presence_logits"], presence
    )
    positive = presence > 0.5
    if positive.any():
        corner_loss = F.smooth_l1_loss(
            output["corners"][positive], target["corners"][positive], beta=0.02
        )
        type_loss = F.cross_entropy(
            output["type_logits"][positive], target["document_type"][positive]
        )
        heatmaps = output["corner_heatmaps"][positive]
        _, _, heatmap_height, heatmap_width = heatmaps.shape
        clipped = target["corners"][positive].clamp(0.0, 1.0)
        target_x = (clipped[..., 0] * (heatmap_width - 1)).round().long()
        target_y = (clipped[..., 1] * (heatmap_height - 1)).round().long()
        target_locations = target_y * heatmap_width + target_x
        heatmap_loss = F.cross_entropy(
            heatmaps.reshape(-1, heatmap_height * heatmap_width),
            target_locations.reshape(-1),
        )
    else:
        corner_loss = output["corners"].sum() * 0.0
        type_loss = output["type_logits"].sum() * 0.0
        heatmap_loss = output["corner_heatmaps"].sum() * 0.0
    quality_loss = F.binary_cross_entropy_with_logits(
        output["quality_logits"],
        target["quality"].to(dtype=output["quality_logits"].dtype),
    )
    total = (
        presence_loss
        + corner_weight * corner_loss
        + type_weight * type_loss
        + quality_weight * quality_loss
        + heatmap_weight * heatmap_loss
    )
    return {
        "total": total,
        "presence": presence_loss,
        "corners": corner_loss,
        "corner_heatmap": heatmap_loss,
        "document_type": type_loss,
        "quality": quality_loss,
    }
