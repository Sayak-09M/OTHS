from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from pii_id.inference.mixed_scanner import VISUAL_CLASS_BY_ENTITY
from pii_id.models.document import MODEL_ARCHITECTURE

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _inside_root(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("readiness input escapes project root")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    if path.suffix.casefold() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _contract_config(value: dict[str, Any]) -> dict[str, str]:
    if value.get("schema_version") != "1.0.0":
        raise ValueError("unsupported DetectionGodX integration config")
    if value.get("producer_role") != "A3":
        raise ValueError("model integration must use DetectionGodX A3")
    if value.get("output_contract") != "dgx.observation.v1":
        raise ValueError("A3 output must be dgx.observation.v1")
    digests = value.get("contract_digests")
    if not isinstance(digests, dict) or not digests:
        raise ValueError("integration config has no contract digests")
    normalized = {str(key): str(item) for key, item in digests.items()}
    if any(_SHA256.fullmatch(item) is None for item in normalized.values()):
        raise ValueError("integration contract digest is invalid")
    return normalized


def assess_readiness(
    *,
    root: Path,
    checkpoint_path: Path = Path("artifacts/document/foundation-v2-best.pt"),
    policy_path: Path = Path("configs/inference/mixed-identities.yaml"),
    validation_report_path: Path = Path(
        "artifacts/reports/foundation-v2-validation.json"
    ),
    integration_config_path: Path = Path(
        "configs/integration/detectiongodx-v2.yaml"
    ),
) -> dict[str, Any]:
    """Verify the local candidate bundle without claiming production accuracy."""

    root = root.resolve(strict=True)
    checkpoint_file = _inside_root(root, checkpoint_path)
    policy_file = _inside_root(root, policy_path)
    validation_file = _inside_root(root, validation_report_path)
    integration_file = _inside_root(root, integration_config_path)

    checkpoint_digest = _sha256(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain an object")
    if checkpoint.get("format") != "pii-id-document-checkpoint":
        raise ValueError("unsupported checkpoint format")
    if checkpoint.get("architecture") != MODEL_ARCHITECTURE:
        raise ValueError("checkpoint architecture does not match the runtime")
    labels_raw = checkpoint.get("label_classes")
    if not isinstance(labels_raw, (list, tuple)):
        raise ValueError("checkpoint has no label classes")
    labels = {str(label) for label in labels_raw}

    policy = _mapping(policy_file)
    if policy.get("schema_version") != "1.0.0":
        raise ValueError("unsupported inference policy")
    detectors = policy.get("detectors")
    if not isinstance(detectors, dict) or not detectors:
        raise ValueError("inference policy has no detectors")
    enabled_entities = sorted(
        str(entity)
        for entity, settings in detectors.items()
        if isinstance(settings, dict) and settings.get("enabled") is True
    )
    missing_visual_labels = sorted(
        entity
        for entity in enabled_entities
        if VISUAL_CLASS_BY_ENTITY.get(entity, entity) not in labels
    )

    validation = _mapping(validation_file)
    if validation.get("format") != "pii-id-document-validation-report":
        raise ValueError("unsupported validation report")
    if validation.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("validation report checkpoint digest does not match")
    metrics = validation.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("overall"), dict):
        raise ValueError("validation report has no overall metrics")

    integration = _mapping(integration_file)
    contract_digests = _contract_config(integration)
    status = "candidate_ready" if not missing_visual_labels else "incomplete"
    tesseract = shutil.which("tesseract")
    return {
        "format": "pii-id-dgx-readiness",
        "format_version": "1.0.0",
        "status": status,
        "production_qualified": False,
        "producer_role": "A3",
        "output_contract": "dgx.observation.v1",
        "software_repository": integration.get("software_repository"),
        "software_commit": integration.get("software_commit"),
        "checkpoint": str(checkpoint_file.relative_to(root)),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_architecture": checkpoint["architecture"],
        "policy": str(policy_file.relative_to(root)),
        "policy_sha256": _sha256(policy_file),
        "validation_report": str(validation_file.relative_to(root)),
        "validation_report_sha256": _sha256(validation_file),
        "validation_metrics": metrics["overall"],
        "enabled_entities": enabled_entities,
        "missing_visual_labels": missing_visual_labels,
        "contract_digests": contract_digests,
        "ocr_executable": tesseract,
        "ocr_available": tesseract is not None,
        "required_before_production": [
            "SIGNED_RUNTIME_MANIFEST_SBOM_AND_LICENSE_BOM",
            "WINDOWS_AND_LINUX_CPU_PARITY",
            "HELD_OUT_REAL_HOSPITAL_AND_HOTEL_PILOT",
            "PER_ENTITY_PRECISION_RECALL_AND_FP_PER_10000",
            "A4_SLICE_CALIBRATION_AND_ASSURANCE_POLICY",
            "AUTHORIZED_PRODUCTION_DATASET_COVERAGE",
        ],
    }
