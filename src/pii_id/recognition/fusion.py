from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .validators import mask_identifier


@dataclass(frozen=True)
class FittedThreshold:
    threshold: float
    precision: float
    recall: float
    enabled: bool


def fit_precision_constrained_threshold(
    *,
    scores: Sequence[float],
    labels: Sequence[bool],
    target_precision: float,
) -> FittedThreshold:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and aligned")
    if not 0.0 < target_precision <= 1.0:
        raise ValueError("target_precision must be in (0, 1]")
    if any(not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("scores must be in [0, 1]")
    positive_count = sum(labels)
    if positive_count == 0:
        return FittedThreshold(1.0, 0.0, 0.0, False)
    candidates: list[FittedThreshold] = []
    for threshold in sorted(set(scores), reverse=True):
        selected = [index for index, score in enumerate(scores) if score >= threshold]
        true_positive = sum(labels[index] for index in selected)
        precision = true_positive / len(selected)
        recall = true_positive / positive_count
        if precision >= target_precision:
            candidates.append(FittedThreshold(threshold, precision, recall, True))
    if not candidates:
        return FittedThreshold(1.0, 0.0, 0.0, False)
    return max(candidates, key=lambda item: (item.recall, -item.threshold))


@dataclass(frozen=True)
class ConfirmationPolicy:
    id_type: str
    clean_threshold: float
    degraded_threshold: float
    enabled: bool
    minimum_keyword_probability: float = 0.80
    minimum_ocr_confidence: float = 0.80
    minimum_ocr_passes: int = 2


@dataclass(frozen=True)
class CandidateEvidence:
    id_type: str
    raw_value: str
    polygon: tuple[tuple[int, int], ...]
    visual_probability: float
    keyword_probability: float
    ocr_confidence: float
    quality_bucket: str
    checksum_valid: bool
    cryptographic_valid: bool
    ocr_pass_count: int = 1


@dataclass(frozen=True)
class ConfirmedFinding:
    id_type: str
    masked_value: str
    polygon: tuple[tuple[int, int], ...]
    status: str
    evidence: tuple[str, ...]


def candidate_rejection_reasons(
    candidate: CandidateEvidence, policy: ConfirmationPolicy
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not policy.enabled:
        reasons.append("policy_disabled")
    if candidate.id_type != policy.id_type:
        reasons.append("type_mismatch")
    threshold = (
        policy.degraded_threshold
        if candidate.quality_bucket == "degraded"
        else policy.clean_threshold
    )
    if candidate.visual_probability < threshold:
        reasons.append("visual_probability_below_threshold")
    if candidate.ocr_confidence < policy.minimum_ocr_confidence:
        reasons.append("ocr_confidence_below_threshold")
    if not candidate.checksum_valid:
        reasons.append("checksum_invalid")
    if (
        candidate.keyword_probability < policy.minimum_keyword_probability
        and not candidate.cryptographic_valid
    ):
        reasons.append("document_context_missing")
    if (
        candidate.ocr_pass_count < policy.minimum_ocr_passes
        and not candidate.cryptographic_valid
    ):
        reasons.append("ocr_pass_agreement_below_threshold")
    return tuple(reasons)


def fuse_identity_candidate(
    candidate: CandidateEvidence, policy: ConfirmationPolicy
) -> ConfirmedFinding | None:
    if candidate_rejection_reasons(candidate, policy):
        return None
    threshold = (
        policy.degraded_threshold
        if candidate.quality_bucket == "degraded"
        else policy.clean_threshold
    )
    evidence = (
        f"visual_type_probability={candidate.visual_probability:.4f}",
        f"ocr_confidence={candidate.ocr_confidence:.4f}",
        f"ocr_pass_agreement={candidate.ocr_pass_count}",
        "checksum=valid",
        (
            "cryptographic_signature=valid"
            if candidate.cryptographic_valid
            else f"keyword_probability={candidate.keyword_probability:.4f}"
        ),
        f"quality_bucket={candidate.quality_bucket}",
        f"calibrated_threshold={threshold:.4f}",
    )
    return ConfirmedFinding(
        id_type=candidate.id_type,
        masked_value=mask_identifier(candidate.raw_value),
        polygon=candidate.polygon,
        status="confirmed",
        evidence=evidence,
    )
