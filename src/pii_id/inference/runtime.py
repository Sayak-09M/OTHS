from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from pii_id.models.document import MODEL_ARCHITECTURE, DocumentPerceptionNet
from pii_id.recognition.ocr import TesseractOcr

from .mixed_scanner import IdentityPolicy, MixedIdentityScanner


@dataclass(frozen=True)
class LoadedScanner:
    scanner: MixedIdentityScanner
    checkpoint_sha256: str
    checkpoint_metrics: Mapping[str, float]


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda_unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps_unavailable")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_policies(path: Path) -> dict[str, IdentityPolicy]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0.0":
        raise ValueError("unsupported inference policy")
    policies_raw = raw.get("detectors")
    if not isinstance(policies_raw, dict):
        raise ValueError("inference policy has no detectors")
    policies: dict[str, IdentityPolicy] = {}
    for id_type, values in policies_raw.items():
        if not isinstance(id_type, str) or not isinstance(values, dict):
            raise ValueError("invalid inference policy entry")
        policies[id_type] = IdentityPolicy(id_type=id_type, **values)
    return policies


def load_scanner(
    *,
    checkpoint_path: Path,
    policy_path: Path,
    tesseract_path: Path | None = None,
    device_name: str = "auto",
) -> LoadedScanner:
    checkpoint = checkpoint_path.expanduser().resolve(strict=True)
    policy = policy_path.expanduser().resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "pii-id-document-checkpoint"
    ):
        raise ValueError("unsupported checkpoint format")
    if payload.get("architecture") != MODEL_ARCHITECTURE:
        raise ValueError(
            "checkpoint architecture is outdated; retrain with train_document.py"
        )
    labels = payload.get("label_classes")
    state_dict = payload.get("state_dict")
    if not isinstance(labels, (list, tuple)) or not isinstance(state_dict, dict):
        raise ValueError("incomplete checkpoint")
    label_classes = tuple(str(label) for label in labels)
    model = DocumentPerceptionNet(
        num_document_types=len(label_classes), pretrained=False
    )
    model.load_state_dict(state_dict, strict=True)
    policies = _load_policies(policy)
    if tesseract_path is None:
        executable = shutil.which("tesseract")
        if executable is None:
            raise RuntimeError("tesseract_not_found")
        tesseract_path = Path(executable)
    device = select_device(device_name)
    scanner = MixedIdentityScanner(
        model=model,
        label_classes=label_classes,
        ocr=TesseractOcr(executable=tesseract_path),
        policies=policies,
        device=device,
    )
    metrics = payload.get("metrics")
    safe_metrics = metrics if isinstance(metrics, dict) else {}
    return LoadedScanner(scanner, _sha256(checkpoint), safe_metrics)
