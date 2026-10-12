from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from .identity_profiles import IDENTITY_PROFILES, IdentityProfile, profile_for
from .ocr import OcrToken

Polygon = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class IdentityCandidate:
    id_type: str
    country: str
    raw_value: str
    polygon: Polygon
    ocr_confidence: float
    context_probability: float
    format_valid: bool
    checksum_valid: bool | None
    source: str
    ocr_pass_count: int = 1
    visual_type_score: float | None = None
    visual_presence_score: float | None = None
    quality_codes: tuple[str, ...] = ()
    validation_codes: tuple[str, ...] = ()


_DETECTORS = tuple(IDENTITY_PROFILES.values())

_OCR_REPAIR_SHAPES = {
    "IN_PAN": "LLLLLDDDDL",
    "IN_EPIC": "LLLDDDDDDD",
    "IN_PASSPORT": "LDDDDDDD",
    "IN_DRIVING_LICENCE": "LL" + "D" * 13,
}
_OCR_REPAIR_LABELS = {
    "IN_PAN": ("identifier", "permanent account", "pan", "income tax"),
    "IN_EPIC": ("identifier", "epic", "voter", "elector"),
    "IN_PASSPORT": ("identifier", "passport"),
    "IN_DRIVING_LICENCE": (
        "identifier",
        "driving licence",
        "driving license",
        "licence no",
        "license no",
        "dl no",
    ),
}
_OCR_TO_DIGIT = {
    "B": "8",
    "D": "0",
    "G": "6",
    "I": "1",
    "L": "1",
    "O": "0",
    "Q": "0",
    "S": "5",
    "Z": "2",
}
_OCR_TO_LETTER = {
    "0": "O",
    "2": "Z",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "$": "S",
}
_NUMERIC_CONTEXT_GATED_ENTITIES = {"IN_ABHA", "NP_NATIONAL_ID", "BD_NID"}
_NEGATIVE_FIELD_LABELS = (
    "date of birth",
    "birth date",
    "dob",
    "issue date",
    "valid until",
    "valid till",
    "mobile",
    "phone",
    "pin code",
    "pincode",
)
_POSITIVE_ID_FIELD_LABELS = {
    "IN_ABHA": ("identifier", "abha", "health id"),
    "NP_NATIONAL_ID": ("identifier", "national id", "national identity"),
    "BD_NID": ("identifier", "nid", "national id", "national identity"),
}


def tokens_from_text(text: str, *, source: str = "native-text") -> tuple[OcrToken, ...]:
    tokens: list[OcrToken] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in re.finditer(r"\S+", line):
            tokens.append(
                OcrToken(
                    text=match.group(0),
                    confidence=1.0,
                    polygon=(
                        (match.start(), line_number),
                        (match.end(), line_number),
                        (match.end(), line_number + 1),
                        (match.start(), line_number + 1),
                    ),
                    line_id=(1, 1, 1, line_number),
                    source=source,
                )
            )
    return tuple(tokens)


def _context_probability(document_text: str, detector: IdentityProfile) -> float:
    folded = f" {document_text.casefold()} "
    if any(keyword.casefold() in folded for keyword in detector.keywords):
        return 0.98
    return 0.0


def _line_text(tokens: Sequence[OcrToken]) -> tuple[str, tuple[tuple[int, int], ...]]:
    pieces: list[str] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for index, token in enumerate(tokens):
        if index:
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(token.text)
        cursor += len(token.text)
        ranges.append((start, cursor))
    return "".join(pieces), tuple(ranges)


def _overlapping_tokens(
    tokens: Sequence[OcrToken], ranges: Sequence[tuple[int, int]], start: int, stop: int
) -> tuple[OcrToken, ...]:
    return tuple(
        token
        for token, (token_start, token_stop) in zip(tokens, ranges, strict=True)
        if token_start < stop and token_stop > start
    )


def _bounding_polygon(tokens: Sequence[OcrToken]) -> Polygon:
    xs = [point[0] for token in tokens for point in token.polygon]
    ys = [point[1] for token in tokens for point in token.polygon]
    return (
        (min(xs), min(ys)),
        (max(xs), min(ys)),
        (max(xs), max(ys)),
        (min(xs), max(ys)),
    )


def normalize_identity_value(id_type: str, value: str) -> str:
    return profile_for(id_type).normalize(value)


def validate_identity_format(id_type: str, value: str) -> bool:
    """Validate a normalized value against the detector's discovery format."""

    try:
        profile = profile_for(id_type)
    except ValueError:
        return False
    return profile.validate(value).format_valid


def _candidate_position(candidate: IdentityCandidate) -> tuple[float, float]:
    return (
        min(point[1] for point in candidate.polygon),
        min(point[0] for point in candidate.polygon),
    )


def _repair_ocr_value(id_type: str, raw_value: str) -> str | None:
    shape = _OCR_REPAIR_SHAPES.get(id_type)
    if shape is None:
        return None
    uppercase = raw_value.upper()
    compact = "".join(
        character
        for character in uppercase
        if character.isascii() and (character.isalnum() or character == "$")
    )
    normalization_cost = int(compact != uppercase)
    if len(compact) == len(shape):
        variants = ((compact, normalization_cost),)
    elif len(compact) == len(shape) + 1:
        variants = tuple(
            (
                compact[:index] + compact[index + 1 :],
                normalization_cost + 2,
            )
            for index in range(len(compact))
        )
    else:
        return None

    results: dict[str, int] = {}
    for value, edit_cost in variants:
        repaired: list[str] = []
        cost = edit_cost
        for character, expected in zip(value, shape, strict=True):
            if expected == "L":
                if "A" <= character <= "Z":
                    repaired.append(character)
                    continue
                replacement = _OCR_TO_LETTER.get(character)
                if character == "1" and id_type != "IN_PASSPORT":
                    replacement = "I"
            else:
                if character.isdigit():
                    repaired.append(character)
                    continue
                replacement = _OCR_TO_DIGIT.get(character)
            if replacement is None:
                break
            repaired.append(replacement)
            cost += 1
        else:
            candidate = "".join(repaired)
            if cost == 0:
                continue
            validation = profile_for(id_type).validate(candidate)
            if validation.format_valid:
                results[candidate] = min(cost, results.get(candidate, cost))
    if not results:
        return None
    minimum_cost = min(results.values())
    best = [value for value, cost in results.items() if cost == minimum_cost]
    return best[0] if len(best) == 1 else None


def _bounds(candidate: IdentityCandidate) -> tuple[float, float, float, float]:
    xs = [point[0] for point in candidate.polygon]
    ys = [point[1] for point in candidate.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _same_ocr_occurrence(
    left: IdentityCandidate, right: IdentityCandidate
) -> bool:
    if (
        left.id_type != right.id_type
        or left.source == "native-text"
        or right.source == "native-text"
    ):
        return False
    left_x0, left_y0, left_x1, left_y1 = _bounds(left)
    right_x0, right_y0, right_x1, right_y1 = _bounds(right)
    intersection_width = max(0.0, min(left_x1, right_x1) - max(left_x0, right_x0))
    intersection_height = max(0.0, min(left_y1, right_y1) - max(left_y0, right_y0))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x1 - left_x0) * max(0.0, left_y1 - left_y0)
    right_area = max(0.0, right_x1 - right_x0) * max(0.0, right_y1 - right_y0)
    return intersection / max(1.0, min(left_area, right_area)) >= 0.60


def _mark_competing_ocr_values(
    candidates: Sequence[IdentityCandidate],
) -> tuple[IdentityCandidate, ...]:
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            if _same_ocr_occurrence(left, candidates[right_index]):
                union(left_index, right_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(find(index), []).append(index)
    quality_additions: dict[int, str] = {}
    for indices in groups.values():
        if len({candidates[index].raw_value for index in indices}) < 2:
            continue
        ranked = sorted(
            indices,
            key=lambda index: (
                candidates[index].checksum_valid is True,
                candidates[index].ocr_pass_count,
                "RECTIFIED_OCR" in candidates[index].validation_codes,
                candidates[index].ocr_confidence,
            ),
            reverse=True,
        )
        top = candidates[ranked[0]]
        second = candidates[ranked[1]]
        dominant = (
            top.checksum_valid is True and second.checksum_valid is not True
        ) or top.ocr_pass_count > second.ocr_pass_count or (
            "RECTIFIED_OCR" in top.validation_codes
            and "RECTIFIED_OCR" not in second.validation_codes
        )
        if dominant:
            for index in ranked[1:]:
                quality_additions[index] = "OCR_VALUE_SUPERSEDED"
        else:
            for index in ranked:
                quality_additions[index] = "OCR_VALUE_CONFLICT"
    return tuple(
        replace(
            candidate,
            quality_codes=tuple(
                sorted(
                    {
                        *candidate.quality_codes,
                        quality_additions[index],
                    }
                )
            ),
        )
        if index in quality_additions
        else candidate
        for index, candidate in enumerate(candidates)
    )


def extract_identity_candidates(
    tokens: Sequence[OcrToken],
    *,
    contextless_entities: frozenset[str] = frozenset(),
) -> tuple[IdentityCandidate, ...]:
    lines: dict[tuple[int, int, int, int], list[OcrToken]] = {}
    for token in tokens:
        lines.setdefault(token.line_id, []).append(token)
    ordered_lines = tuple(
        tuple(sorted(line_tokens, key=lambda item: item.polygon[0][0]))
        for line_tokens in lines.values()
    )
    document_lines = tuple(_line_text(line_tokens)[0] for line_tokens in ordered_lines)
    document_text = " ".join(document_lines)
    contexts = {
        detector.id_type: _context_probability(document_text, detector)
        for detector in _DETECTORS
    }
    candidates: list[IdentityCandidate] = []
    for line_index, ordered in enumerate(ordered_lines):
        line, ranges = _line_text(ordered)
        uppercase = line.upper()
        local_context = line
        if line_index:
            local_context = f"{document_lines[line_index - 1]} {line}"
        local_context_folded = local_context.casefold()
        for detector in _DETECTORS:
            context = contexts[detector.id_type]
            if (
                detector.context_required_for_discovery
                and context == 0.0
                and detector.id_type not in contextless_entities
            ):
                continue
            if detector.id_type in _NUMERIC_CONTEXT_GATED_ENTITIES:
                has_negative_label = any(
                    label in local_context_folded for label in _NEGATIVE_FIELD_LABELS
                )
                has_id_label = any(
                    label in local_context_folded
                    for label in _POSITIVE_ID_FIELD_LABELS[detector.id_type]
                )
                if has_negative_label and not has_id_label:
                    continue
            for match in detector.discovery_pattern.finditer(uppercase):
                matched_tokens = _overlapping_tokens(
                    ordered, ranges, match.start(), match.end()
                )
                if not matched_tokens:
                    continue
                raw_value = detector.normalize(match.group(0))
                validation = detector.validate(
                    raw_value, document_lines=document_lines
                )
                candidates.append(
                    IdentityCandidate(
                        id_type=detector.id_type,
                        country=detector.country,
                        raw_value=raw_value,
                        polygon=_bounding_polygon(matched_tokens),
                        ocr_confidence=sum(
                            token.confidence for token in matched_tokens
                        )
                        / len(matched_tokens),
                        context_probability=context,
                        format_valid=validation.format_valid,
                        checksum_valid=validation.checksum_valid,
                        source=matched_tokens[0].source,
                        validation_codes=validation.signal_codes,
                    )
                )
        folded_line = local_context_folded
        for detector in _DETECTORS:
            labels = _OCR_REPAIR_LABELS.get(detector.id_type)
            if labels is None or not any(label in folded_line for label in labels):
                continue
            context = contexts[detector.id_type]
            if context == 0.0 and detector.id_type not in contextless_entities:
                continue
            for start in range(len(ordered)):
                for stop in range(start + 1, min(len(ordered), start + 4) + 1):
                    token_span = ordered[start:stop]
                    if any(token.source == "native-text" for token in token_span):
                        continue
                    repaired_value = _repair_ocr_value(
                        detector.id_type,
                        " ".join(token.text for token in token_span),
                    )
                    if repaired_value is None:
                        continue
                    validation = detector.validate(
                        repaired_value, document_lines=document_lines
                    )
                    candidate = IdentityCandidate(
                        id_type=detector.id_type,
                        country=detector.country,
                        raw_value=repaired_value,
                        polygon=_bounding_polygon(token_span),
                        ocr_confidence=sum(
                            token.confidence for token in token_span
                        )
                        / len(token_span),
                        context_probability=context,
                        format_valid=validation.format_valid,
                        checksum_valid=validation.checksum_valid,
                        source=token_span[0].source,
                        validation_codes=tuple(
                            sorted(
                                {
                                    *validation.signal_codes,
                                    "OCR_POSITIONAL_REPAIR",
                                }
                            )
                        ),
                    )
                    if not any(
                        existing.id_type == candidate.id_type
                        and existing.raw_value == candidate.raw_value
                        and existing.polygon == candidate.polygon
                        for existing in candidates
                    ):
                        candidates.append(candidate)
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.id_type,
                candidate.raw_value,
                *_candidate_position(candidate),
            ),
        )
    )


def merge_identity_passes(
    passes: Iterable[Sequence[IdentityCandidate]],
) -> tuple[IdentityCandidate, ...]:
    grouped: dict[tuple[str, str], list[list[IdentityCandidate]]] = {}
    for candidates in passes:
        occurrences_by_value: dict[tuple[str, str], list[IdentityCandidate]] = {}
        for candidate in candidates:
            key = (candidate.id_type, candidate.raw_value)
            occurrences_by_value.setdefault(key, []).append(candidate)
        for key, occurrences in occurrences_by_value.items():
            ordered = sorted(occurrences, key=_candidate_position)
            slots = grouped.setdefault(key, [])
            while len(slots) < len(ordered):
                slots.append([])
            for index, candidate in enumerate(ordered):
                slots[index].append(candidate)
    merged: list[IdentityCandidate] = []
    for key in sorted(grouped):
        for candidates in grouped[key]:
            strongest = max(candidates, key=lambda item: item.ocr_confidence)
            checksum_values = [
                item.checksum_valid
                for item in candidates
                if item.checksum_valid is not None
            ]
            checksum_valid = (
                False
                if False in checksum_values
                else True
                if True in checksum_values
                else None
            )
            merged.append(
                replace(
                    strongest,
                    context_probability=max(
                        item.context_probability for item in candidates
                    ),
                    format_valid=all(item.format_valid for item in candidates),
                    checksum_valid=checksum_valid,
                    validation_codes=tuple(
                        sorted(
                            {
                                code
                                for item in candidates
                                for code in item.validation_codes
                            }
                        )
                    ),
                    ocr_pass_count=len(candidates),
                )
            )
    return _mark_competing_ocr_values(merged)
