from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from .ocr import OcrToken
from .validators import normalize_decimal_digits, validate_aadhaar

_AADHAAR_KEYWORDS = (
    "aadhaar",
    "aadhar",
    "uidai",
    "unique identification",
    "आधार",
    "আধার",
    "આધાર",
    "ਆਧਾਰ",
    "ಆಧಾರ್",
    "ആധാർ",
    "ଆଧାର",
    "ஆதார்",
    "ఆధార్",
)
_INDIA_CONTEXT = ("government of india", "भारत सरकार", "भारत")


@dataclass(frozen=True)
class AadhaarCandidate:
    raw_value: str
    polygon: tuple[tuple[int, int], ...]
    ocr_confidence: float
    keyword_probability: float
    checksum_valid: bool
    ocr_pass_count: int = 1


def _context_probability(tokens: Sequence[OcrToken]) -> float:
    document_text = " ".join(token.text for token in tokens).casefold()
    if any(keyword.casefold() in document_text for keyword in _AADHAAR_KEYWORDS):
        return 0.98
    if any(keyword.casefold() in document_text for keyword in _INDIA_CONTEXT):
        return 0.82
    return 0.0


def _candidate_characters_only(value: str) -> bool:
    return all(character.isdecimal() or character in " -" for character in value)


def _bounding_polygon(tokens: Sequence[OcrToken]) -> tuple[tuple[int, int], ...]:
    xs = [point[0] for token in tokens for point in token.polygon]
    ys = [point[1] for token in tokens for point in token.polygon]
    return (
        (min(xs), min(ys)),
        (max(xs), min(ys)),
        (max(xs), max(ys)),
        (min(xs), max(ys)),
    )


def extract_aadhaar_candidates(
    tokens: Sequence[OcrToken],
) -> tuple[AadhaarCandidate, ...]:
    context_probability = _context_probability(tokens)
    lines: dict[tuple[int, int, int, int], list[OcrToken]] = {}
    for item in tokens:
        lines.setdefault(item.line_id, []).append(item)
    candidates: dict[str, AadhaarCandidate] = {}
    for line_tokens in lines.values():
        ordered = sorted(line_tokens, key=lambda item: item.polygon[0][0])
        for start in range(len(ordered)):
            for stop in range(start + 1, min(len(ordered), start + 4) + 1):
                span = ordered[start:stop]
                combined = " ".join(item.text for item in span)
                if not _candidate_characters_only(combined):
                    continue
                digits = normalize_decimal_digits(combined)
                if len(digits) != 12:
                    continue
                confidence = sum(item.confidence for item in span) / len(span)
                candidate = AadhaarCandidate(
                    raw_value=digits,
                    polygon=_bounding_polygon(span),
                    ocr_confidence=confidence,
                    keyword_probability=context_probability,
                    checksum_valid=validate_aadhaar(digits),
                )
                previous = candidates.get(digits)
                is_better = (
                    previous is None
                    or candidate.ocr_confidence > previous.ocr_confidence
                )
                if is_better:
                    candidates[digits] = candidate
    return tuple(candidates[key] for key in sorted(candidates))


def merge_aadhaar_passes(
    passes: Sequence[Sequence[AadhaarCandidate]],
) -> tuple[AadhaarCandidate, ...]:
    grouped: dict[str, list[AadhaarCandidate]] = {}
    for candidates in passes:
        seen_in_pass: set[str] = set()
        for candidate in candidates:
            if candidate.raw_value in seen_in_pass:
                continue
            seen_in_pass.add(candidate.raw_value)
            grouped.setdefault(candidate.raw_value, []).append(candidate)
    merged: list[AadhaarCandidate] = []
    for raw_value in sorted(grouped):
        candidates = grouped[raw_value]
        strongest = max(candidates, key=lambda item: item.ocr_confidence)
        merged.append(
            replace(
                strongest,
                keyword_probability=max(
                    candidate.keyword_probability for candidate in candidates
                ),
                checksum_valid=all(
                    candidate.checksum_valid for candidate in candidates
                ),
                ocr_pass_count=len(candidates),
            )
        )
    return tuple(merged)
