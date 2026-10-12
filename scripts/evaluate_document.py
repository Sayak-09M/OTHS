#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pii_id.data.loader import DatasetManifest  # noqa: E402
from pii_id.data.torch_dataset import (  # noqa: E402
    DocumentLabelSpace,
    TorchDocumentDataset,
)
from pii_id.evaluation.access import (  # noqa: E402
    AccessPurpose,
    SplitAccessGuard,
)
from pii_id.evaluation.document_report import (  # noqa: E402
    summarize_document_predictions,
)
from pii_id.evaluation.metrics import polygon_iou  # noqa: E402
from pii_id.models.document import DocumentPerceptionNet  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a document checkpoint")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "artifacts/document/best.pt"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/reports/document-validation.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within_root(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(ROOT):
        raise ValueError("path escapes project root")
    return resolved


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    options = arguments()
    checkpoint_path = within_root(options.checkpoint).resolve(strict=True)
    output_path = within_root(options.output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "pii-id-document-checkpoint":
        raise ValueError("unsupported checkpoint format")
    label_classes = tuple(checkpoint["label_classes"])
    config = checkpoint["config"]
    artifact_root = within_root(ROOT / str(config["artifact_root"]))
    guard = SplitAccessGuard(
        root=ROOT, ledger_path=artifact_root / "audit" / "dataset-access.jsonl"
    )
    grant = guard.authorize(
        split="validation",
        purpose=AccessPurpose.MODEL_SELECTION,
        run_id=f"document-report-{sha256(checkpoint_path)[:16]}",
    )
    manifest_path = within_root(ROOT / str(config["validation_manifest"]))
    manifest = DatasetManifest(
        root=ROOT,
        manifest_path=manifest_path,
        registry_path=within_root(ROOT / str(config["registry"])),
    )
    records = list(manifest.iter_records())
    label_space = DocumentLabelSpace(classes=label_classes)
    dataset = TorchDocumentDataset(
        manifest=manifest,
        grant=grant,
        label_space=label_space,
        input_size=int(config["input_size"]),
    )
    loader = DataLoader(
        dataset, batch_size=options.batch_size, shuffle=False, num_workers=0
    )
    model = DocumentPerceptionNet(
        num_document_types=len(label_classes), pretrained=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = select_device()
    model.to(device).eval()
    rows: list[dict[str, object]] = []
    record_index = 0
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(device))
            predicted_presence = (
                torch.sigmoid(output["presence_logits"]).cpu() >= 0.5
            ).tolist()
            predicted_types = output["type_logits"].argmax(dim=1).cpu().tolist()
            predicted_corners = output["corners"].cpu().tolist()
            truth_presence = (batch["presence"] > 0.5).tolist()
            truth_types = batch["document_type"].tolist()
            truth_corners = batch["corners"].tolist()
            for (
                is_present,
                is_predicted_present,
                truth_type,
                predicted_type,
                truth_polygon,
                predicted_polygon,
            ) in zip(
                truth_presence,
                predicted_presence,
                truth_types,
                predicted_types,
                truth_corners,
                predicted_corners,
                strict=True,
            ):
                record = records[record_index]
                rows.append(
                    {
                        "truth_type": (
                            label_classes[truth_type] if is_present else "NON_ID"
                        ),
                        "predicted_type": label_classes[predicted_type],
                        "truth_presence": is_present,
                        "predicted_presence": is_predicted_present,
                        "polygon_iou": (
                            polygon_iou(truth_polygon, predicted_polygon)
                            if is_present
                            else 0.0
                        ),
                        "country": record["country"],
                        "script": record["script"],
                        "modality": record["modality"],
                        "dataset_id": record["dataset_id"],
                    }
                )
                record_index += 1
    report = {
        "format": "pii-id-document-validation-report",
        "format_version": "1.0.0",
        "checkpoint_sha256": sha256(checkpoint_path),
        "manifest_sha256": sha256(manifest_path),
        "split": "validation",
        "device": str(device),
        "metrics": summarize_document_predictions(rows),
    }
    write_json_atomic(output_path, report)
    print(
        json.dumps(
            {
                "report": str(output_path.relative_to(ROOT)),
                **report["metrics"]["overall"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
