#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pii_id.augmentation.degradations import DegradationAugmenter  # noqa: E402
from pii_id.data.loader import DatasetManifest  # noqa: E402
from pii_id.data.torch_dataset import (  # noqa: E402
    DocumentLabelSpace,
    TorchDocumentDataset,
)
from pii_id.evaluation.access import (  # noqa: E402
    AccessPurpose,
    SplitAccessGuard,
)
from pii_id.models.document import DocumentPerceptionNet  # noqa: E402
from pii_id.training.document_trainer import (  # noqa: E402
    evaluate_document_model,
    save_document_checkpoint,
    train_document_epoch,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the document perception model")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/detection/baseline.yaml"
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help="Bound each split for a pipeline smoke run",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Avoid downloading ImageNet weights (intended for smoke tests)",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(ROOT):
        raise ValueError("configuration must be inside the project root")
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "1.0.0":
        raise ValueError("unsupported training configuration")
    return config


def project_path(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("configuration path must be a string")
    path = (ROOT / value).resolve(strict=False)
    if not path.is_relative_to(ROOT):
        raise ValueError("configuration path escapes the project root")
    return path


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    artifact_root = project_path(config["artifact_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    guard = SplitAccessGuard(
        root=ROOT, ledger_path=artifact_root / "audit" / "dataset-access.jsonl"
    )
    train_grant = guard.authorize(
        split="train", purpose=AccessPurpose.FIT, run_id=f"document-fit-{seed}"
    )
    validation_grant = guard.authorize(
        split="validation",
        purpose=AccessPurpose.MODEL_SELECTION,
        run_id=f"document-select-{seed}",
    )
    registry = project_path(config["registry"])
    train_manifest = DatasetManifest(
        root=ROOT,
        manifest_path=project_path(config["train_manifest"]),
        registry_path=registry,
    )
    validation_manifest = DatasetManifest(
        root=ROOT,
        manifest_path=project_path(config["validation_manifest"]),
        registry_path=registry,
    )
    label_space = DocumentLabelSpace.from_records(train_manifest.iter_records())
    input_size = int(config["input_size"])
    train_dataset = TorchDocumentDataset(
        manifest=train_manifest,
        grant=train_grant,
        label_space=label_space,
        input_size=input_size,
        augmenter=DegradationAugmenter(seed=seed),
    )
    validation_dataset = TorchDocumentDataset(
        manifest=validation_manifest,
        grant=validation_grant,
        label_space=label_space,
        input_size=input_size,
    )
    if arguments.smoke_limit is not None:
        if arguments.smoke_limit <= 0:
            raise ValueError("smoke-limit must be positive")
        train_dataset = Subset(
            train_dataset, range(min(arguments.smoke_limit, len(train_dataset)))
        )
        validation_dataset = Subset(
            validation_dataset,
            range(min(arguments.smoke_limit, len(validation_dataset))),
        )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )
    pretrained = bool(config["pretrained"]) and not arguments.no_pretrained
    model = DocumentPerceptionNet(
        num_document_types=len(label_space.classes), pretrained=pretrained
    )
    device = select_device()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = arguments.epochs or int(config["epochs"])
    best_score = float("-inf")
    for epoch in range(1, epochs + 1):
        base_train_dataset = (
            train_dataset.dataset
            if isinstance(train_dataset, Subset)
            else train_dataset
        )
        base_train_dataset.set_epoch(epoch)
        training = train_document_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device
        )
        validation = evaluate_document_model(
            model=model, loader=validation_loader, device=device
        )
        score = (
            validation["mean_polygon_iou"]
            + validation["document_type_accuracy"]
            + validation["presence_accuracy"]
            - validation["false_positive_rate"]
        )
        event = {
            "epoch": epoch,
            "device": str(device),
            "training": training,
            "validation": validation,
        }
        print(json.dumps(event, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            checkpoint = save_document_checkpoint(
                model=model,
                artifact_root=artifact_root,
                relative_path=Path(str(config["checkpoint"])),
                label_classes=label_space.classes,
                config={
                    **config,
                    "pretrained": pretrained,
                    "epochs_executed": epoch,
                    "smoke_limit": arguments.smoke_limit,
                },
                metrics=validation,
            )
            print(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint.path.relative_to(ROOT)),
                        "sha256": checkpoint.sha256,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
