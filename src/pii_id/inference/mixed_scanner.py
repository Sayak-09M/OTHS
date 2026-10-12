from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import torch
from PIL import Image, ImageEnhance, ImageOps

from pii_id.geometry import letterbox_document
from pii_id.models.document import QUALITY_LABELS
from pii_id.recognition.identity_profiles import (
    IDENTITY_PROFILES,
    validate_identity_value,
)
from pii_id.recognition.mixed_ids import (
    IdentityCandidate,
    extract_identity_candidates,
    merge_identity_passes,
    tokens_from_text,
    validate_identity_format,
)
from pii_id.recognition.ocr import OcrEngineError, OcrToken
from pii_id.recognition.validators import mask_identifier

from .rectification import RectificationTransform, rectify_document

_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
VISUAL_CLASS_BY_ENTITY = {
    id_type: profile.visual_class or id_type
    for id_type, profile in IDENTITY_PROFILES.items()
}


class OcrProtocol(Protocol):
    def recognize(
        self, image: Image.Image, *, scripts: Sequence[str]
    ) -> tuple[OcrToken, ...]: ...


@dataclass(frozen=True)
class IdentityPolicy:
    id_type: str
    enabled: bool
    require_checksum: bool
    minimum_context_probability: float
    minimum_ocr_confidence: float
    minimum_ocr_passes: int
    minimum_visual_type_score: float = 0.80
    strong_consensus_ocr_passes: int = 3
    strong_consensus_minimum_ocr_confidence: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "minimum_context_probability",
            "minimum_ocr_confidence",
            "minimum_visual_type_score",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.strong_consensus_minimum_ocr_confidence is not None and not (
            0.0 <= self.strong_consensus_minimum_ocr_confidence <= 1.0
        ):
            raise ValueError(
                "strong_consensus_minimum_ocr_confidence must be in [0, 1]"
            )
        if self.minimum_ocr_passes < 1 or self.strong_consensus_ocr_passes < 1:
            raise ValueError("OCR pass requirements must be positive")


@dataclass(frozen=True)
class MixedFinding:
    id_type: str
    country: str
    masked_value: str
    polygon: tuple[tuple[float, float], ...]
    status: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ReviewCandidate:
    id_type: str
    country: str
    masked_value: str
    polygon: tuple[tuple[float, float], ...]
    rejection_reasons: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class MixedEvent:
    code: str
    stage: str
    data: Mapping[str, object]

    def as_log(self) -> dict[str, object]:
        return {"code": self.code, "stage": self.stage, "data": dict(self.data)}


@dataclass(frozen=True)
class MixedScanResult:
    findings: tuple[MixedFinding, ...]
    review_candidates: tuple[ReviewCandidate, ...]
    events: tuple[MixedEvent, ...]


@dataclass(frozen=True)
class CandidateScanResult:
    """Raw candidates for an in-process A3 adapter; never serialize directly."""

    candidates: tuple[IdentityCandidate, ...]
    events: tuple[MixedEvent, ...]


@dataclass(frozen=True)
class _VisualAnalysis:
    event: MixedEvent
    presence_score: float
    type_scores: Mapping[str, float]
    quality_codes: tuple[str, ...]
    source_corners: tuple[tuple[float, float], ...]


def _ocr_variants(
    image: Image.Image,
) -> tuple[tuple[Image.Image, tuple[str, ...]], ...]:
    original = image.convert("RGB")
    grayscale = ImageOps.autocontrast(ImageOps.grayscale(original)).convert("RGB")
    sharpened = ImageEnhance.Sharpness(grayscale).enhance(2.0)
    return (
        (original, ("latin",)),
        (
            grayscale,
            ("latin", "devanagari", "bengali", "nepali", "tamil", "sinhala"),
        ),
        (sharpened, ("latin",)),
    )


def _rejection_reasons(
    candidate: IdentityCandidate, policy: IdentityPolicy | None
) -> tuple[str, ...]:
    if policy is None or not policy.enabled:
        return ("detector_disabled",)
    reasons: list[str] = []
    if any(
        code in {"OCR_VALUE_CONFLICT", "OCR_VALUE_SUPERSEDED"}
        for code in candidate.quality_codes
    ):
        reasons.append("competing_ocr_value")
    if not candidate.format_valid:
        reasons.append("format_invalid")
    if candidate.checksum_valid is False or (
        policy.require_checksum and candidate.checksum_valid is not True
    ):
        reasons.append("checksum_invalid")
    if _context_source(candidate, policy) == "missing":
        reasons.append("document_context_missing")
    minimum_confidence = policy.minimum_ocr_confidence
    if (
        candidate.ocr_pass_count >= policy.strong_consensus_ocr_passes
        and policy.strong_consensus_minimum_ocr_confidence is not None
    ):
        minimum_confidence = policy.strong_consensus_minimum_ocr_confidence
    if candidate.ocr_confidence < minimum_confidence:
        reasons.append("ocr_confidence_below_threshold")
    native_text = candidate.source == "native-text"
    if (
        not native_text
        and candidate.checksum_valid is not True
        and candidate.ocr_pass_count < policy.minimum_ocr_passes
    ):
        reasons.append("ocr_pass_agreement_below_threshold")
    return tuple(reasons)


def _context_source(candidate: IdentityCandidate, policy: IdentityPolicy) -> str:
    if candidate.context_probability >= policy.minimum_context_probability:
        return "text"
    if (
        candidate.visual_type_score is not None
        and candidate.visual_presence_score is not None
        and candidate.visual_type_score >= policy.minimum_visual_type_score
        and candidate.visual_presence_score >= policy.minimum_visual_type_score
    ):
        return "visual"
    return "missing"


def _evidence(
    candidate: IdentityCandidate, policy: IdentityPolicy | None
) -> tuple[str, ...]:
    evidence = [
        f"ocr_confidence={candidate.ocr_confidence:.4f}",
        f"context_probability={candidate.context_probability:.4f}",
        f"source_passes={candidate.ocr_pass_count}",
        f"format_valid={str(candidate.format_valid).lower()}",
    ]
    if policy is not None:
        evidence.append(f"context_source={_context_source(candidate, policy)}")
        required_confidence = policy.minimum_ocr_confidence
        if (
            candidate.ocr_pass_count >= policy.strong_consensus_ocr_passes
            and policy.strong_consensus_minimum_ocr_confidence is not None
        ):
            required_confidence = policy.strong_consensus_minimum_ocr_confidence
        evidence.append(f"required_ocr_confidence={required_confidence:.4f}")
    if candidate.source == "native-text":
        evidence.append("native_text")
    if candidate.checksum_valid is not None:
        evidence.append(f"checksum_valid={str(candidate.checksum_valid).lower()}")
    if candidate.visual_type_score is not None:
        evidence.append(f"visual_type_score={candidate.visual_type_score:.6f}")
    if candidate.visual_presence_score is not None:
        evidence.append(
            f"visual_presence_score={candidate.visual_presence_score:.6f}"
        )
    evidence.extend(f"validator={code}" for code in candidate.validation_codes)
    evidence.extend(f"quality={code}" for code in candidate.quality_codes)
    return tuple(evidence)


class MixedIdentityScanner:
    """Document-agnostic ID discovery; visual classification is never an OCR gate."""

    def __init__(
        self,
        *,
        ocr: OcrProtocol,
        policies: Mapping[str, IdentityPolicy],
        model: torch.nn.Module | None = None,
        label_classes: tuple[str, ...] = (),
        device: torch.device | None = None,
        input_size: int = 224,
        rectified_size: tuple[int, int] = (1024, 640),
        minimum_rectification_score: float = 0.50,
    ) -> None:
        if not 0.0 <= minimum_rectification_score <= 1.0:
            raise ValueError("rectification score must be in [0, 1]")
        if rectified_size[0] < 2 or rectified_size[1] < 2:
            raise ValueError("rectified output must be at least 2x2")
        self.ocr = ocr
        self.policies = dict(policies)
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval() if model is not None else None
        self.label_classes = label_classes
        self.input_size = input_size
        self.rectified_size = rectified_size
        self.minimum_rectification_score = minimum_rectification_score

    def candidate_rejection_reasons(
        self, candidate: IdentityCandidate
    ) -> tuple[str, ...]:
        """Return the active policy decision without exposing the candidate."""

        return _rejection_reasons(candidate, self.policies.get(candidate.id_type))

    def reference_value_rejection_reasons(
        self, id_type: str, value: str
    ) -> tuple[str, ...]:
        """Validate labelled truth without applying OCR confidence thresholds."""

        policy = self.policies.get(id_type)
        if policy is None or not policy.enabled:
            return ("detector_disabled",)
        reasons: list[str] = []
        if not validate_identity_format(id_type, value):
            reasons.append("format_invalid")
        validation = validate_identity_value(id_type, value)
        if validation.checksum_valid is False or (
            policy.require_checksum and validation.checksum_valid is not True
        ):
            reasons.append("checksum_invalid")
        return tuple(reasons)

    @torch.inference_mode()
    def _visual_analysis(self, image: Image.Image) -> _VisualAnalysis | None:
        if self.model is None:
            return None
        full_polygon = [
            [0, 0],
            [image.width - 1, 0],
            [image.width - 1, image.height - 1],
            [0, image.height - 1],
        ]
        tensor, _, letterbox = letterbox_document(
            image, full_polygon, size=self.input_size
        )
        normalized = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
        output = self.model(normalized.unsqueeze(0).to(self.device))
        probabilities = torch.softmax(output["type_logits"], dim=1)[0].cpu()
        type_index = int(probabilities.argmax())
        top_type = (
            self.label_classes[type_index]
            if type_index < len(self.label_classes)
            else "UNKNOWN"
        )
        type_scores = {
            label: float(probabilities[index])
            for index, label in enumerate(self.label_classes)
            if index < len(probabilities)
        }
        presence_score = float(torch.sigmoid(output["presence_logits"])[0].cpu())
        quality_probabilities = torch.sigmoid(output["quality_logits"])[0].cpu()
        quality_codes = tuple(
            label.upper()
            for label, score in zip(
                QUALITY_LABELS, quality_probabilities.tolist(), strict=True
            )
            if score >= 0.5
        )
        normalized_corners = output["corners"][0].cpu().tolist()
        source_corners = letterbox.input_to_source(
            normalized_corners, normalized=True
        )
        clamped_corners = tuple(
            (
                min(max(float(x), 0.0), float(image.width - 1)),
                min(max(float(y), 0.0), float(image.height - 1)),
            )
            for x, y in source_corners
        )
        event = MixedEvent(
            "visual_support",
            "visual",
            {
                "top_type": top_type,
                "top_score": round(float(probabilities[type_index]), 6),
                "presence_score": round(presence_score, 6),
                "quality_codes": quality_codes,
                "used_as_gate": False,
            },
        )
        return _VisualAnalysis(
            event=event,
            presence_score=presence_score,
            type_scores=type_scores,
            quality_codes=quality_codes,
            source_corners=clamped_corners,
        )

    def _recognize_variants(
        self,
        image: Image.Image,
        *,
        transform_to_source: RectificationTransform | None = None,
        contextless_entities: frozenset[str] = frozenset(),
    ) -> tuple[list[tuple[IdentityCandidate, ...]], int]:
        passes: list[tuple[IdentityCandidate, ...]] = []
        failures = 0
        for variant, scripts in _ocr_variants(image):
            try:
                tokens = self.ocr.recognize(variant, scripts=scripts)
            except OcrEngineError:
                failures += 1
                continue
            candidates = extract_identity_candidates(
                tokens, contextless_entities=contextless_entities
            )
            route_code = (
                "RECTIFIED_OCR"
                if transform_to_source is not None
                else "FULL_PAGE_OCR"
            )
            candidates = tuple(
                replace(
                    candidate,
                    validation_codes=tuple(
                        sorted({*candidate.validation_codes, route_code})
                    ),
                )
                for candidate in candidates
            )
            if transform_to_source is not None:
                candidates = tuple(
                    replace(
                        candidate,
                        polygon=tuple(
                            (float(x), float(y))
                            for x, y in transform_to_source.rectified_to_source(
                                candidate.polygon
                            )
                        ),
                    )
                    for candidate in candidates
                )
            passes.append(candidates)
        return passes, failures

    def _evaluate(
        self, candidates: Sequence[IdentityCandidate], events: list[MixedEvent]
    ) -> MixedScanResult:
        findings: list[MixedFinding] = []
        reviews: list[ReviewCandidate] = []
        rejection_counts: dict[str, int] = {}
        for candidate in candidates:
            policy = self.policies.get(candidate.id_type)
            reasons = _rejection_reasons(candidate, policy)
            evidence = _evidence(candidate, policy)
            if reasons:
                for reason in reasons:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                reviews.append(
                    ReviewCandidate(
                        id_type=candidate.id_type,
                        country=candidate.country,
                        masked_value=mask_identifier(candidate.raw_value),
                        polygon=candidate.polygon,
                        rejection_reasons=reasons,
                        evidence=evidence,
                    )
                )
                continue
            status = (
                "checksum_and_context_validated"
                if candidate.checksum_valid is True
                else "format_and_context_validated"
            )
            findings.append(
                MixedFinding(
                    id_type=candidate.id_type,
                    country=candidate.country,
                    masked_value=mask_identifier(candidate.raw_value),
                    polygon=candidate.polygon,
                    status=status,
                    evidence=evidence,
                )
            )
        events.append(
            MixedEvent(
                "identity_candidates",
                "validation",
                {
                    "candidates": len(candidates),
                    "findings": len(findings),
                    "review_candidates": len(reviews),
                    "rejections": rejection_counts,
                },
            )
        )
        return MixedScanResult(tuple(findings), tuple(reviews), tuple(events))

    def scan_text_candidates(self, text: str) -> CandidateScanResult:
        tokens = tokens_from_text(text)
        candidates = extract_identity_candidates(tokens)
        events = [
            MixedEvent(
                "native_text_processed",
                "extraction",
                {"tokens": len(tokens)},
            )
        ]
        return CandidateScanResult(candidates, tuple(events))

    def scan_text(self, text: str) -> MixedScanResult:
        candidate_result = self.scan_text_candidates(text)
        return self._evaluate(
            candidate_result.candidates, list(candidate_result.events)
        )

    def scan_image_candidates(
        self, image: Image.Image, *, native_text: str | None = None
    ) -> CandidateScanResult:
        events: list[MixedEvent] = []
        try:
            visual = self._visual_analysis(image)
        except (IndexError, KeyError, RuntimeError, TypeError, ValueError):
            visual = None
            events.append(
                MixedEvent(
                    "visual_inference_failed",
                    "visual",
                    {"full_page_ocr_preserved": True},
                )
            )
        if visual is not None:
            events.append(visual.event)
        passes: list[tuple[IdentityCandidate, ...]] = []
        failed_passes = 0
        contextless_entities = frozenset(
            id_type
            for id_type, policy in self.policies.items()
            if visual is not None
            and policy.enabled
            and visual.presence_score >= policy.minimum_visual_type_score
            and visual.type_scores.get(
                VISUAL_CLASS_BY_ENTITY.get(id_type, id_type), 0.0
            )
            >= policy.minimum_visual_type_score
        )
        if native_text and native_text.strip():
            passes.append(
                extract_identity_candidates(
                    tokens_from_text(native_text),
                    contextless_entities=contextless_entities,
                )
            )

        full_passes, full_failures = self._recognize_variants(
            image, contextless_entities=contextless_entities
        )
        passes.extend(full_passes)
        failed_passes += full_failures

        rectified_ocr_used = False
        if (
            visual is not None
            and visual.presence_score >= self.minimum_rectification_score
        ):
            try:
                rectified, transform = rectify_document(
                    image,
                    visual.source_corners,
                    output_size=self.rectified_size,
                )
            except ValueError:
                events.append(
                    MixedEvent(
                        "visual_rectification_failed",
                        "rectification",
                        {"full_page_ocr_preserved": True},
                    )
                )
            else:
                rectified_passes, rectified_failures = self._recognize_variants(
                    rectified,
                    transform_to_source=transform,
                    contextless_entities=contextless_entities,
                )
                passes.extend(rectified_passes)
                failed_passes += rectified_failures
                rectified_ocr_used = True

        candidates = merge_identity_passes(passes)
        if visual is not None:
            candidates = tuple(
                replace(
                    candidate,
                    visual_type_score=visual.type_scores.get(
                        VISUAL_CLASS_BY_ENTITY.get(
                            candidate.id_type, candidate.id_type
                        )
                    ),
                    visual_presence_score=visual.presence_score,
                    quality_codes=tuple(
                        sorted({*candidate.quality_codes, *visual.quality_codes})
                    ),
                )
                for candidate in candidates
            )
        events.append(
            MixedEvent(
                "content_extracted",
                "extraction",
                {
                    "successful_sources": len(passes),
                    "failed_ocr_passes": failed_passes,
                    "candidate_count": len(candidates),
                    "native_text_used": bool(native_text and native_text.strip()),
                    "rectified_ocr_used": rectified_ocr_used,
                },
            )
        )
        return CandidateScanResult(candidates, tuple(events))

    def scan_image(
        self, image: Image.Image, *, native_text: str | None = None
    ) -> MixedScanResult:
        candidate_result = self.scan_image_candidates(
            image, native_text=native_text
        )
        return self._evaluate(
            candidate_result.candidates, list(candidate_result.events)
        )
