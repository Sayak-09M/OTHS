from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pii_id.recognition.mixed_ids import (
    IdentityCandidate,
    normalize_identity_value,
)
from pii_id.recognition.validators import mask_identifier

ENTITY_BY_DOCUMENT_TYPE = {
    "IN_AADHAAR": "IN_AADHAAR",
    "IN_PAN": "IN_PAN",
    "IN_EPIC": "IN_EPIC",
    "IN_ABHA": "IN_ABHA",
    "IN_PASSPORT": "IN_PASSPORT",
    "IN_DRIVING_LICENCE": "IN_DRIVING_LICENCE",
    "NP_ID_CARD": "NP_NATIONAL_ID",
    "BD_NID": "BD_NID",
    "LK_NIC": "LK_NIC",
}


@dataclass(frozen=True, order=True)
class IdentityKey:
    id_type: str
    value: str


@dataclass(frozen=True)
class IdentityEvaluationRow:
    manifest_line: int
    dataset_id: str
    identity_group: str
    split: str
    document_type: str
    document_entity: str | None
    country: str
    script: str
    modality: str
    selected_entities: tuple[str, ...]
    score_kind: str
    truth: frozenset[IdentityKey]
    confirmable_truth: frozenset[IdentityKey]
    truth_rejection_reasons: Mapping[IdentityKey, tuple[str, ...]]
    candidates: frozenset[IdentityKey]
    confirmed: frozenset[IdentityKey]
    rejection_reasons: Mapping[IdentityKey, tuple[str, ...]]
    event_codes: tuple[str, ...]
    failed_ocr_passes: int


def _truth_identities(
    record: Mapping[str, Any], selected_entities: frozenset[str]
) -> frozenset[IdentityKey]:
    document_type = record.get("document_type")
    entity = (
        ENTITY_BY_DOCUMENT_TYPE.get(document_type)
        if isinstance(document_type, str)
        else None
    )
    if entity not in selected_entities:
        return frozenset()
    annotations = record.get("annotations")
    fields = annotations.get("fields") if isinstance(annotations, Mapping) else None
    if not isinstance(fields, list):
        return frozenset()
    truth: set[IdentityKey] = set()
    for field in fields:
        if not isinstance(field, Mapping) or field.get("canonical_name") != "id_number":
            continue
        value = field.get("value")
        if value is None:
            continue
        normalized = normalize_identity_value(entity, str(value))
        if normalized:
            truth.add(IdentityKey(entity, normalized))
    return frozenset(truth)


def evaluate_identity_record(
    record: Mapping[str, Any],
    *,
    candidates: Sequence[IdentityCandidate],
    selected_entities: set[str] | frozenset[str],
    rejection_reasons: Callable[[IdentityCandidate], Sequence[str]],
    truth_rejection_reasons: Callable[[str, str], Sequence[str]] | None = None,
    manifest_line: int = 0,
    event_codes: Sequence[str] = (),
    failed_ocr_passes: int = 0,
) -> IdentityEvaluationRow:
    """Evaluate one image while keeping exact identifiers in memory only."""

    selected = frozenset(selected_entities)
    if not selected:
        raise ValueError("at least one entity must be selected")
    if manifest_line < 0:
        raise ValueError("manifest line cannot be negative")
    if failed_ocr_passes < 0:
        raise ValueError("failed OCR pass count cannot be negative")
    document_type = str(record.get("document_type", "UNKNOWN"))
    document_entity = ENTITY_BY_DOCUMENT_TYPE.get(document_type)
    truth = _truth_identities(record, selected)
    truth_rejections = {
        key: tuple(truth_rejection_reasons(key.id_type, key.value))
        for key in truth
        if truth_rejection_reasons is not None
    }
    truth_rejections = {
        key: reasons for key, reasons in truth_rejections.items() if reasons
    }
    confirmable_truth = frozenset(truth - set(truth_rejections))

    discovered: set[IdentityKey] = set()
    confirmed: set[IdentityKey] = set()
    rejected: dict[IdentityKey, set[str]] = {}
    for candidate in candidates:
        if candidate.id_type not in selected:
            continue
        key = IdentityKey(
            candidate.id_type,
            normalize_identity_value(candidate.id_type, candidate.raw_value),
        )
        if not key.value:
            continue
        discovered.add(key)
        reasons = tuple(rejection_reasons(candidate))
        if reasons:
            rejected.setdefault(key, set()).update(reasons)
        else:
            confirmed.add(key)

    # If one occurrence was confirmed, an independently rejected duplicate does
    # not make the normalized identity itself rejected.
    for key in confirmed:
        rejected.pop(key, None)

    if truth:
        score_kind = "exact"
    elif document_entity in selected:
        score_kind = "weak_positive"
    else:
        score_kind = "negative"

    return IdentityEvaluationRow(
        manifest_line=manifest_line,
        dataset_id=str(record.get("dataset_id", "unknown")),
        identity_group=str(record.get("identity_group", "unknown")),
        split=str(record.get("split", "unknown")),
        document_type=document_type,
        document_entity=document_entity,
        country=str(record.get("country", "unknown")),
        script=str(record.get("script", "unknown")),
        modality=str(record.get("modality", "unknown")),
        selected_entities=tuple(sorted(selected)),
        score_kind=score_kind,
        truth=truth,
        confirmable_truth=confirmable_truth,
        truth_rejection_reasons=truth_rejections,
        candidates=frozenset(discovered),
        confirmed=frozenset(confirmed),
        rejection_reasons={
            key: tuple(sorted(reasons)) for key, reasons in rejected.items()
        },
        event_codes=tuple(event_codes),
        failed_ocr_passes=failed_ocr_passes,
    )


def _metrics(
    rows: Sequence[IdentityEvaluationRow],
    *,
    level: str,
    entity: str | None = None,
) -> dict[str, float | int]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    exact_match_records = 0
    positive_records = 0
    negative_records = 0
    true_negative_records = 0
    false_positive_records = 0
    scored_records = 0
    for row in rows:
        if row.score_kind == "weak_positive":
            continue
        if level == "candidate":
            truth = row.truth
            predictions = row.candidates
        else:
            truth = row.confirmable_truth
            predictions = row.confirmed
        if entity is not None:
            truth = frozenset(item for item in truth if item.id_type == entity)
            predictions = frozenset(
                item for item in predictions if item.id_type == entity
            )
        scored_records += 1
        true_positives += len(truth & predictions)
        false_positives += len(predictions - truth)
        false_negatives += len(truth - predictions)
        exact_match_records += truth == predictions
        if truth:
            positive_records += 1
        else:
            negative_records += 1
            if predictions:
                false_positive_records += 1
            else:
                true_negative_records += 1

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "scored_records": scored_records,
        "positive_records": positive_records,
        "negative_records": negative_records,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_records": exact_match_records,
        "exact_match_rate": exact_match_records / max(1, scored_records),
        "true_negative_records": true_negative_records,
        "false_positive_records": false_positive_records,
        "false_positive_records_per_10k": (
            false_positive_records / max(1, negative_records) * 10_000
        ),
    }


def _slice_metrics(
    rows: Sequence[IdentityEvaluationRow], attribute: str
) -> dict[str, dict[str, float | int]]:
    values = sorted({str(getattr(row, attribute)) for row in rows})
    return {
        value: _metrics(
            [row for row in rows if str(getattr(row, attribute)) == value],
            level="confirmed",
        )
        for value in values
    }


def _masked(key: IdentityKey) -> dict[str, str]:
    return {
        "id_type": key.id_type,
        "masked_value": mask_identifier(key.value, keep_last=4),
    }


def _failure(row: IdentityEvaluationRow) -> dict[str, object] | None:
    truth_status: list[dict[str, object]] = []
    for truth in sorted(row.truth):
        truth_reasons = row.truth_rejection_reasons.get(truth, ())
        if truth_reasons:
            status = "excluded_invalid_ground_truth"
            reasons = truth_reasons
        elif truth in row.confirmed:
            status = "confirmed"
            reasons: tuple[str, ...] = ()
        elif truth in row.candidates:
            status = "rejected_by_policy"
            reasons = row.rejection_reasons.get(truth, ())
        else:
            status = "not_extracted"
            reasons = ()
        item: dict[str, object] = {**_masked(truth), "status": status}
        if reasons:
            item["rejection_reasons"] = reasons
        truth_status.append(item)

    unexpected = sorted(row.confirmed - row.confirmable_truth)
    candidate_only = sorted(row.candidates - row.confirmed - row.truth)
    weak_missing = (
        row.score_kind == "weak_positive"
        and row.document_entity is not None
        and not any(
            item.id_type == row.document_entity for item in row.confirmed
        )
    )
    candidate_failure = row.score_kind != "weak_positive" and (
        row.truth != row.candidates
    )
    confirmed_failure = row.score_kind != "weak_positive" and (
        row.confirmable_truth != row.confirmed
    )
    if not candidate_failure and not confirmed_failure and not weak_missing:
        return None
    failure: dict[str, object] = {
        "manifest_line": row.manifest_line,
        "dataset_id": row.dataset_id,
        "split": row.split,
        "document_type": row.document_type,
        "script": row.script,
        "modality": row.modality,
        "score_kind": row.score_kind,
        "truth_status": truth_status,
        "unexpected_confirmed": [_masked(item) for item in unexpected],
        "candidate_only": [_masked(item) for item in candidate_only],
        "event_codes": row.event_codes,
        "failed_ocr_passes": row.failed_ocr_passes,
    }
    if weak_missing:
        failure["diagnostic"] = (
            "no_same_type_confirmed_candidate_without_exact_ground_truth"
        )
    return failure


def summarize_identity_evaluation(
    rows: Sequence[IdentityEvaluationRow],
) -> dict[str, object]:
    selected_entities = sorted(
        {entity for row in rows for entity in row.selected_entities}
    )
    weak_rows = [row for row in rows if row.score_kind == "weak_positive"]
    weak_summary: dict[str, dict[str, float | int]] = {}
    for entity in selected_entities:
        entity_rows = [row for row in weak_rows if row.document_entity == entity]
        if not entity_rows:
            continue
        candidate_hits = sum(
            any(item.id_type == entity for item in row.candidates)
            for row in entity_rows
        )
        confirmed_hits = sum(
            any(item.id_type == entity for item in row.confirmed)
            for row in entity_rows
        )
        weak_summary[entity] = {
            "records": len(entity_rows),
            "candidate_hits": candidate_hits,
            "candidate_rate": candidate_hits / len(entity_rows),
            "confirmed_hits": confirmed_hits,
            "confirmed_rate": confirmed_hits / len(entity_rows),
        }

    failures = [failure for row in rows if (failure := _failure(row)) is not None]
    ground_truth_issues = [
        {
            "manifest_line": row.manifest_line,
            "dataset_id": row.dataset_id,
            "document_type": row.document_type,
            "modality": row.modality,
            **_masked(key),
            "validation_reasons": reasons,
        }
        for row in rows
        for key, reasons in sorted(row.truth_rejection_reasons.items())
    ]
    exact_rows = [row for row in rows if row.score_kind != "weak_positive"]
    return {
        "selected_entities": selected_entities,
        "coverage": {
            "records_processed": len(rows),
            "exact_records": len(exact_rows),
            "exact_positive_records": sum(bool(row.truth) for row in exact_rows),
            "confirmable_positive_records": sum(
                bool(row.confirmable_truth) for row in exact_rows
            ),
            "negative_records": sum(not row.truth for row in exact_rows),
            "weak_positive_records": len(weak_rows),
            "invalid_truth_identifiers": len(ground_truth_issues),
            "unique_identity_groups": len({row.identity_group for row in rows}),
        },
        "candidate_level": _metrics(rows, level="candidate"),
        "confirmed_level": _metrics(rows, level="confirmed"),
        "by_entity": {
            entity: _metrics(rows, level="confirmed", entity=entity)
            for entity in selected_entities
        },
        "by_dataset": _slice_metrics(rows, "dataset_id"),
        "by_modality": _slice_metrics(rows, "modality"),
        "by_script": _slice_metrics(rows, "script"),
        "weak_positive_detection": weak_summary,
        "runtime": {
            "failed_ocr_passes": sum(row.failed_ocr_passes for row in rows),
            "event_counts": {
                code: sum(code in row.event_codes for row in rows)
                for code in sorted({code for row in rows for code in row.event_codes})
            },
        },
        "ground_truth_issues": ground_truth_issues,
        "failures": failures,
    }
