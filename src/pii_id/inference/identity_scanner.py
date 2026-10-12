from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from PIL import Image, ImageEnhance, ImageOps

from pii_id.geometry import letterbox_document
from pii_id.recognition.candidates import (
    extract_aadhaar_candidates,
    merge_aadhaar_passes,
)
from pii_id.recognition.fusion import (
    CandidateEvidence,
    ConfirmationPolicy,
    candidate_rejection_reasons,
    fuse_identity_candidate,
)
from pii_id.recognition.ocr import OcrEngineError, OcrToken

from .rectification import rectify_document

_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


class OcrProtocol(Protocol):
    def recognize(
        self, image: Image.Image, *, scripts: Sequence[str]
    ) -> tuple[OcrToken, ...]: ...


@dataclass(frozen=True)
class InferenceEvent:
    code: str
    stage: str
    counts: Mapping[str, int]

    def as_log(self) -> dict[str, object]:
        return {"code": self.code, "stage": self.stage, "counts": dict(self.counts)}


@dataclass(frozen=True)
class ImageFinding:
    id_type: str
    masked_value: str
    source_polygon: tuple[tuple[float, float], ...]
    status: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ImageScanResult:
    findings: tuple[ImageFinding, ...]
    events: tuple[InferenceEvent, ...]


def _ocr_variants(image: Image.Image) -> tuple[Image.Image, ...]:
    original = image.convert("RGB")
    grayscale = ImageOps.autocontrast(ImageOps.grayscale(original)).convert("RGB")
    sharpened = ImageEnhance.Sharpness(grayscale).enhance(2.0)
    return original, grayscale, sharpened


class IdentityDocumentScanner:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        label_classes: tuple[str, ...],
        ocr: OcrProtocol,
        policies: Mapping[str, ConfirmationPolicy],
        device: torch.device,
        input_size: int = 224,
        rectified_size: tuple[int, int] = (1024, 640),
        minimum_presence_probability: float = 0.90,
    ) -> None:
        if not 0.0 < minimum_presence_probability < 1.0:
            raise ValueError("presence probability must be in (0, 1)")
        self.model = model.to(device).eval()
        self.label_classes = label_classes
        self.ocr = ocr
        self.policies = dict(policies)
        self.device = device
        self.input_size = input_size
        self.rectified_size = rectified_size
        self.minimum_presence_probability = minimum_presence_probability

    @torch.inference_mode()
    def scan_image(self, image: Image.Image) -> ImageScanResult:
        events: list[InferenceEvent] = []
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
        presence_probability = float(torch.sigmoid(output["presence_logits"])[0].cpu())
        if presence_probability < self.minimum_presence_probability:
            events.append(InferenceEvent("no_document", "detection", {}))
            return ImageScanResult((), tuple(events))
        type_probabilities = torch.softmax(output["type_logits"], dim=1)[0].cpu()
        type_index = int(type_probabilities.argmax())
        if type_index >= len(self.label_classes):
            events.append(InferenceEvent("invalid_type_output", "classification", {}))
            return ImageScanResult((), tuple(events))
        id_type = self.label_classes[type_index]
        policy = self.policies.get(id_type)
        if policy is None or not policy.enabled:
            events.append(
                InferenceEvent("unsupported_document_type", "classification", {})
            )
            return ImageScanResult((), tuple(events))
        normalized_corners = output["corners"][0].cpu().tolist()
        source_corners = letterbox.input_to_source(normalized_corners, normalized=True)
        try:
            rectified, rectification = rectify_document(
                image, source_corners, output_size=self.rectified_size
            )
        except ValueError:
            events.append(
                InferenceEvent("invalid_document_geometry", "rectification", {})
            )
            return ImageScanResult((), tuple(events))
        ocr_passes = []
        for variant in _ocr_variants(rectified):
            try:
                tokens = self.ocr.recognize(variant, scripts=("latin",))
            except OcrEngineError:
                events.append(InferenceEvent("ocr_pass_failed", "ocr", {}))
                continue
            ocr_passes.append(extract_aadhaar_candidates(tokens))
        candidates = merge_aadhaar_passes(tuple(ocr_passes))
        events.append(
            InferenceEvent(
                "ocr_candidates",
                "ocr",
                {"passes": len(ocr_passes), "candidates": len(candidates)},
            )
        )
        degraded = bool((torch.sigmoid(output["quality_logits"])[0].cpu() >= 0.5).any())
        findings: list[ImageFinding] = []
        rejection_counts: dict[str, int] = {}
        for candidate in candidates:
            evidence = CandidateEvidence(
                id_type=id_type,
                raw_value=candidate.raw_value,
                polygon=candidate.polygon,
                visual_probability=float(type_probabilities[type_index]),
                keyword_probability=candidate.keyword_probability,
                ocr_confidence=candidate.ocr_confidence,
                quality_bucket="degraded" if degraded else "clean",
                checksum_valid=candidate.checksum_valid,
                cryptographic_valid=False,
                ocr_pass_count=candidate.ocr_pass_count,
            )
            reasons = candidate_rejection_reasons(evidence, policy)
            for reason in reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            fused = fuse_identity_candidate(evidence, policy)
            if fused is None:
                continue
            source_polygon = rectification.rectified_to_source(candidate.polygon)
            findings.append(
                ImageFinding(
                    id_type=fused.id_type,
                    masked_value=fused.masked_value,
                    source_polygon=tuple((x, y) for x, y in source_polygon),
                    status=fused.status,
                    evidence=fused.evidence,
                )
            )
        if rejection_counts:
            events.append(
                InferenceEvent("candidate_rejections", "fusion", rejection_counts)
            )
        events.append(
            InferenceEvent(
                "confirmed_finding" if findings else "no_confirmed_candidate",
                "fusion",
                {"findings": len(findings)},
            )
        )
        return ImageScanResult(tuple(findings), tuple(events))
