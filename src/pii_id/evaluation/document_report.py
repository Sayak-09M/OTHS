from __future__ import annotations

from collections.abc import Mapping, Sequence

from .metrics import classification_metrics


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    count = len(rows)
    positives = [row for row in rows if bool(row["truth_presence"])]
    negatives = [row for row in rows if not bool(row["truth_presence"])]
    positive_count = len(positives)
    negative_count = len(negatives)
    return {
        "examples": count,
        "positive_examples": positive_count,
        "negative_examples": negative_count,
        "presence_accuracy": sum(
            bool(row["truth_presence"]) == bool(row["predicted_presence"])
            for row in rows
        )
        / max(1, count),
        "false_positive_rate": sum(bool(row["predicted_presence"]) for row in negatives)
        / max(1, negative_count),
        "false_negative_rate": sum(
            not bool(row["predicted_presence"]) for row in positives
        )
        / max(1, positive_count),
        "mean_polygon_iou": sum(float(row["polygon_iou"]) for row in positives)
        / max(1, positive_count),
        "recall_iou_50": sum(float(row["polygon_iou"]) >= 0.50 for row in positives)
        / max(1, positive_count),
        "recall_iou_75": sum(float(row["polygon_iou"]) >= 0.75 for row in positives)
        / max(1, positive_count),
        "document_type_accuracy": sum(
            row["truth_type"] == row["predicted_type"] for row in positives
        )
        / max(1, positive_count),
    }


def _grouped(
    rows: Sequence[Mapping[str, object]], key: str
) -> dict[str, dict[str, float | int]]:
    values = sorted({str(row[key]) for row in rows})
    return {
        value: _summary([row for row in rows if str(row[key]) == value])
        for value in values
    }


def summarize_document_predictions(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("at least one prediction row is required")
    required = {
        "truth_type",
        "predicted_type",
        "truth_presence",
        "predicted_presence",
        "polygon_iou",
        "country",
        "script",
        "modality",
        "dataset_id",
    }
    if any(not required.issubset(row) for row in rows):
        raise ValueError("prediction rows are missing required keys")
    return {
        "overall": _summary(rows),
        "classification": classification_metrics(
            truth=[str(row["truth_type"]) for row in rows if row["truth_presence"]],
            prediction=[
                str(row["predicted_type"]) for row in rows if row["truth_presence"]
            ],
        ),
        "by_document_type": _grouped(rows, "truth_type"),
        "by_country": _grouped(rows, "country"),
        "by_script": _grouped(rows, "script"),
        "by_modality": _grouped(rows, "modality"),
        "by_dataset": _grouped(rows, "dataset_id"),
    }
