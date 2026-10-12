from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from pii_id.evaluation.metrics import polygon_iou
from pii_id.models.document import MODEL_ARCHITECTURE, document_multitask_loss


def _targets_from_batch(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device)
        for key in (
            "presence",
            "corners",
            "document_type",
            "quality",
        )
    }


def train_document_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    example_count = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = _targets_from_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        losses = document_multitask_loss(output, targets)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_size = images.shape[0]
        loss_sum += float(losses["total"].detach()) * batch_size
        example_count += batch_size
    return {"loss": loss_sum / max(1, example_count), "examples": float(example_count)}


@torch.inference_mode()
def evaluate_document_model(
    *, model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    iou_sum = 0.0
    correct_types = 0
    example_count = 0
    positive_count = 0
    negative_count = 0
    correct_presence = 0
    false_positives = 0
    false_negatives = 0
    for batch in loader:
        images = batch["image"].to(device)
        output = model(images)
        truth_presence = batch["presence"].cpu() > 0.5
        predicted_presence = torch.sigmoid(output["presence_logits"]).cpu() >= 0.5
        correct_presence += int((predicted_presence == truth_presence).sum())
        false_positives += int((predicted_presence & ~truth_presence).sum())
        false_negatives += int((~predicted_presence & truth_presence).sum())
        positive_count += int(truth_presence.sum())
        negative_count += int((~truth_presence).sum())
        predicted_types = output["type_logits"].argmax(dim=1).cpu()
        truth_types = batch["document_type"].cpu()
        correct_types += int(((predicted_types == truth_types) & truth_presence).sum())
        predicted_corners = output["corners"].cpu().tolist()
        truth_corners = batch["corners"].cpu().tolist()
        iou_sum += sum(
            polygon_iou(truth, prediction)
            for truth, prediction, positive in zip(
                truth_corners,
                predicted_corners,
                truth_presence.tolist(),
                strict=True,
            )
            if positive
        )
        example_count += images.shape[0]
    return {
        "mean_polygon_iou": iou_sum / max(1, positive_count),
        "document_type_accuracy": correct_types / max(1, positive_count),
        "presence_accuracy": correct_presence / max(1, example_count),
        "false_positive_rate": false_positives / max(1, negative_count),
        "false_negative_rate": false_negatives / max(1, positive_count),
        "positive_examples": float(positive_count),
        "negative_examples": float(negative_count),
        "examples": float(example_count),
    }


@dataclass(frozen=True)
class CheckpointResult:
    path: Path
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_document_checkpoint(
    *,
    model: torch.nn.Module,
    artifact_root: Path,
    relative_path: Path,
    label_classes: tuple[str, ...],
    config: Mapping[str, object],
    metrics: Mapping[str, float],
) -> CheckpointResult:
    root = artifact_root.resolve(strict=True)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("checkpoint path must stay below artifact root")
    target = (root / relative_path).resolve(strict=False)
    if not target.is_relative_to(root) or target == root:
        raise ValueError("checkpoint path must stay below artifact root")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("checkpoint parent escapes artifact root")
    payload = {
        "format": "pii-id-document-checkpoint",
        "format_version": "1.0.0",
        "architecture": MODEL_ARCHITECTURE,
        "state_dict": model.state_dict(),
        "label_classes": label_classes,
        "config": dict(config),
        "metrics": dict(metrics),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return CheckpointResult(path=target, sha256=_sha256(target))
