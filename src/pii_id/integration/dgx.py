from __future__ import annotations

import hashlib
import hmac
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pii_id.inference.mixed_scanner import CandidateScanResult
from pii_id.recognition.mixed_ids import IdentityCandidate
from pii_id.recognition.validators import mask_identifier

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DETECTOR_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_CATEGORY_BY_ENTITY = {
    "IN_AADHAAR": "identity.india",
    "IN_PAN": "identity.india",
    "IN_EPIC": "identity.india",
    "IN_ABHA": "identity.india",
    "IN_PASSPORT": "identity.india",
    "IN_DRIVING_LICENCE": "identity.india",
    "NP_NATIONAL_ID": "identity.national",
    "BD_NID": "identity.national",
    "LK_NIC": "identity.national",
}
_FINGERPRINT_DOMAIN = b"dgx.identity-fingerprint.v1"


def _validate_uuid(value: str, name: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a canonical UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUID")


def _validate_digest(value: str, name: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class DgxEvidenceReference:
    artifact_id: str
    byte_offset: int
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_uuid(self.artifact_id, "evidence.artifact_id")
        _validate_digest(self.sha256, "evidence.sha256")
        if self.byte_offset < 0:
            raise ValueError("evidence.byte_offset must be non-negative")
        if not 1 <= self.byte_length <= 1024 * 1024:
            raise ValueError("evidence.byte_length must be in [1, 1048576]")


@dataclass(frozen=True)
class DgxObservationContext:
    client_id: str
    scan_run_id: str
    work_item_id: str
    generation_id: str
    root_file_version_id: str
    content_node_id: str
    source_artifact_id: str
    fragment_id: str
    policy_digest: str
    pipeline_digest: str
    runtime_manifest_digest: str
    detector_id: str
    detector_digest: str
    model_digest: str
    page: int
    image_width: int
    image_height: int
    evidence: DgxEvidenceReference
    created_at: datetime
    container_ancestry: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "client_id",
            "scan_run_id",
            "work_item_id",
            "generation_id",
            "root_file_version_id",
            "content_node_id",
            "source_artifact_id",
            "fragment_id",
        ):
            _validate_uuid(getattr(self, name), name)
        for name in (
            "policy_digest",
            "pipeline_digest",
            "runtime_manifest_digest",
            "detector_digest",
            "model_digest",
        ):
            _validate_digest(getattr(self, name), name)
        if _DETECTOR_ID.fullmatch(self.detector_id) is None:
            raise ValueError("detector_id is invalid")
        if self.page < 1:
            raise ValueError("page must be positive")
        if self.image_width < 1 or self.image_height < 1:
            raise ValueError("image dimensions must be positive")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if len(self.container_ancestry) > 32:
            raise ValueError("container ancestry exceeds 32 entries")
        for index, (content_node_id, artifact_id) in enumerate(
            self.container_ancestry
        ):
            _validate_uuid(
                content_node_id,
                f"container_ancestry[{index}].content_node_id",
            )
            _validate_uuid(artifact_id, f"container_ancestry[{index}].artifact_id")


def _identity_fingerprint(
    key: bytes, *, entity_type: str, normalized_value: str
) -> str:
    if len(key) < 32:
        raise ValueError("fingerprint key must be at least 32 bytes")
    message = b"\0".join(
        (
            _FINGERPRINT_DOMAIN,
            entity_type.encode("ascii"),
            normalized_value.encode("utf-8"),
        )
    )
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _candidate_validation(candidate: IdentityCandidate) -> str:
    if not candidate.format_valid or candidate.checksum_valid is False:
        return "invalid"
    if candidate.checksum_valid is True:
        return "valid"
    return "indeterminate"


def _uncertainty(candidate: IdentityCandidate, validation: str) -> str:
    if validation == "invalid":
        return "high"
    if (
        validation == "valid"
        and candidate.ocr_confidence >= 0.80
        and candidate.context_probability >= 0.80
        and candidate.ocr_pass_count >= 2
    ):
        return "low"
    if candidate.ocr_confidence < 0.50 or candidate.ocr_pass_count < 2:
        return "high"
    return "medium"


def _signals(candidate: IdentityCandidate) -> list[str]:
    signals = {
        "NATIVE_TEXT_EVIDENCE"
        if candidate.source == "native-text"
        else "OCR_EVIDENCE",
        "FORMAT_VALID" if candidate.format_valid else "FORMAT_INVALID",
        "CONTEXT_MATCH"
        if candidate.context_probability >= 0.80
        else "CONTEXT_MISSING",
    }
    if candidate.ocr_pass_count >= 2:
        signals.add("MULTIPASS_AGREEMENT")
    else:
        signals.add("SINGLE_PASS_ONLY")
    if candidate.checksum_valid is True:
        signals.add("CHECKSUM_VALID")
    elif candidate.checksum_valid is False:
        signals.add("CHECKSUM_INVALID")
    else:
        signals.add("CHECKSUM_NOT_APPLICABLE")
    if candidate.visual_type_score is None:
        signals.add("VISUAL_TYPE_UNAVAILABLE")
    elif candidate.visual_type_score >= 0.50:
        signals.add("VISUAL_TYPE_SUPPORT")
    else:
        signals.add("VISUAL_TYPE_CONFLICT")
    signals.update(f"QUALITY_{code}" for code in candidate.quality_codes)
    signals.update(candidate.validation_codes)
    return sorted(signals)


def _bounding_box(
    candidate: IdentityCandidate, context: DgxObservationContext
) -> list[float]:
    xs = [
        min(max(float(point[0]), 0.0), float(context.image_width))
        for point in candidate.polygon
    ]
    ys = [
        min(max(float(point[1]), 0.0), float(context.image_height))
        for point in candidate.polygon
    ]
    left = min(xs)
    top = min(ys)
    return [left, top, max(xs) - left, max(ys) - top]


def _created_at(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def build_page_observations(
    scan: CandidateScanResult,
    *,
    context: DgxObservationContext,
    fingerprint_key: bytes,
) -> tuple[dict[str, Any], ...]:
    """Convert in-memory page candidates to raw-free ``dgx.observation.v1`` values.

    The caller owns the protected evidence artifact and supplies a client-scoped,
    work-scoped fingerprint key. This A3 adapter emits candidates only; A4 remains
    the sole authority that can publish findings.
    """

    if len(fingerprint_key) < 32:
        raise ValueError("fingerprint key must be at least 32 bytes")
    observations: list[dict[str, Any]] = []
    for candidate in scan.candidates:
        category_id = _CATEGORY_BY_ENTITY.get(candidate.id_type)
        if category_id is None:
            raise ValueError(f"unsupported DetectionGodX entity: {candidate.id_type}")
        masked_value = mask_identifier(candidate.raw_value, keep_last=4)
        if masked_value == candidate.raw_value:
            raise ValueError("masked value must not expose the complete candidate")
        fingerprint = _identity_fingerprint(
            fingerprint_key,
            entity_type=candidate.id_type,
            normalized_value=candidate.raw_value,
        )
        box = _bounding_box(candidate, context)
        occurrence_name = "|".join(
            (
                candidate.id_type,
                fingerprint,
                str(context.page),
                *(f"{coordinate:.6f}" for coordinate in box),
            )
        )
        observation_id = str(
            uuid.uuid5(uuid.UUID(context.work_item_id), occurrence_name)
        )
        validation = _candidate_validation(candidate)
        observations.append(
            {
                "contract": "dgx.observation.v1",
                "observation_id": observation_id,
                "client_id": context.client_id,
                "scan_run_id": context.scan_run_id,
                "work_item_id": context.work_item_id,
                "generation_id": context.generation_id,
                "root_file_version_id": context.root_file_version_id,
                "content_node_id": context.content_node_id,
                "source_artifact_id": context.source_artifact_id,
                "fragment_id": context.fragment_id,
                "policy_digest": context.policy_digest,
                "pipeline_digest": context.pipeline_digest,
                "runtime_manifest_digest": context.runtime_manifest_digest,
                "entity_type": candidate.id_type,
                "category_id": category_id,
                "country": candidate.country,
                "detector": {
                    "detector_id": context.detector_id,
                    "detector_digest": context.detector_digest,
                    "model_digest": context.model_digest,
                },
                "candidate_validation": validation,
                "masked_value": masked_value,
                "candidate_value_length": len(candidate.raw_value),
                "identity_fingerprint": fingerprint,
                "fingerprint_scheme": "hmac-sha256-client-v1",
                # This is the OCR recognizer score, not an asserted calibrated
                # probability. A4 calibrates it per entity/quality slice.
                "detector_score": round(candidate.ocr_confidence, 6),
                "uncertainty": _uncertainty(candidate, validation),
                "source_locator": {
                    "locator_kind": "page_region",
                    "coordinate_unit": "bounding_box",
                    "page": context.page,
                    "bounding_box": box,
                    "coordinate_space": {
                        "origin": "top_left",
                        "unit": "pixel",
                        "rotation_degrees": 0.0,
                        "width": float(context.image_width),
                        "height": float(context.image_height),
                    },
                    "container_ancestry": [
                        {
                            "content_node_id": content_node_id,
                            "artifact_id": artifact_id,
                        }
                        for content_node_id, artifact_id in context.container_ancestry
                    ],
                },
                "evidence_reference": {
                    "artifact_id": context.evidence.artifact_id,
                    "content_class": "finding_evidence",
                    "byte_offset": context.evidence.byte_offset,
                    "byte_length": context.evidence.byte_length,
                    "sha256": context.evidence.sha256,
                    "encryption_required": True,
                },
                "signal_codes": _signals(candidate),
                "created_at": _created_at(context.created_at),
            }
        )
    return tuple(observations)
