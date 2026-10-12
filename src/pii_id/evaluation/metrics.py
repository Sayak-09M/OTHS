from __future__ import annotations

from collections.abc import Sequence

Point = Sequence[float]
Polygon = Sequence[Point]


def _signed_area(polygon: Polygon) -> float:
    return 0.5 * sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )


def _counter_clockwise(polygon: Polygon) -> list[tuple[float, float]]:
    points = [(float(point[0]), float(point[1])) for point in polygon]
    return points if _signed_area(points) >= 0 else list(reversed(points))


def _inside(point: Point, edge_start: Point, edge_end: Point) -> bool:
    cross = (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (
        edge_end[1] - edge_start[1]
    ) * (point[0] - edge_start[0])
    return cross >= -1e-9


def _line_intersection(
    segment_start: Point,
    segment_end: Point,
    edge_start: Point,
    edge_end: Point,
) -> tuple[float, float]:
    x1, y1 = segment_start
    x2, y2 = segment_end
    x3, y3 = edge_start
    x4, y4 = edge_end
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return float(x2), float(y2)
    determinant_a = x1 * y2 - y1 * x2
    determinant_b = x3 * y4 - y3 * x4
    x = (determinant_a * (x3 - x4) - (x1 - x2) * determinant_b) / denominator
    y = (determinant_a * (y3 - y4) - (y1 - y2) * determinant_b) / denominator
    return x, y


def _convex_intersection(subject: Polygon, clip: Polygon) -> list[tuple[float, float]]:
    output = _counter_clockwise(subject)
    clip_points = _counter_clockwise(clip)
    for index, edge_start in enumerate(clip_points):
        edge_end = clip_points[(index + 1) % len(clip_points)]
        input_points = output
        output = []
        if not input_points:
            break
        segment_start = input_points[-1]
        for segment_end in input_points:
            end_inside = _inside(segment_end, edge_start, edge_end)
            start_inside = _inside(segment_start, edge_start, edge_end)
            if end_inside:
                if not start_inside:
                    output.append(
                        _line_intersection(
                            segment_start, segment_end, edge_start, edge_end
                        )
                    )
                output.append(segment_end)
            elif start_inside:
                output.append(
                    _line_intersection(segment_start, segment_end, edge_start, edge_end)
                )
            segment_start = segment_end
    return output


def polygon_iou(truth: Polygon, prediction: Polygon) -> float:
    """Return IoU for two convex polygons, independent of winding direction."""
    if len(truth) < 3 or len(prediction) < 3:
        raise ValueError("polygons must contain at least three points")
    truth_area = abs(_signed_area(truth))
    prediction_area = abs(_signed_area(prediction))
    intersection = _convex_intersection(truth, prediction)
    intersection_area = (
        abs(_signed_area(intersection)) if len(intersection) >= 3 else 0.0
    )
    union_area = truth_area + prediction_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def _edit_distance[Token](
    truth: Sequence[Token], prediction: Sequence[Token]
) -> int:
    previous = list(range(len(prediction) + 1))
    for truth_index, truth_token in enumerate(truth, start=1):
        current = [truth_index]
        for prediction_index, prediction_token in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1]
                    + (truth_token != prediction_token),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(pairs: Sequence[tuple[str, str]]) -> float:
    edits = sum(_edit_distance(truth, prediction) for truth, prediction in pairs)
    truth_characters = sum(len(truth) for truth, _ in pairs)
    return edits / max(1, truth_characters)


def word_error_rate(pairs: Sequence[tuple[str, str]]) -> float:
    edits = sum(
        _edit_distance(truth.split(), prediction.split())
        for truth, prediction in pairs
    )
    truth_words = sum(len(truth.split()) for truth, _ in pairs)
    return edits / max(1, truth_words)


def classification_metrics(
    *, truth: Sequence[str], prediction: Sequence[str]
) -> dict[str, object]:
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction lengths differ")
    labels = sorted(set(truth) | set(prediction))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []
    for label in labels:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(truth, prediction, strict=True)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(truth, prediction, strict=True)
        )
        false_negative = sum(
            actual == label and predicted != label
            for actual, predicted in zip(truth, prediction, strict=True)
        )
        support = sum(actual == label for actual in truth)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_scores.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    correct = sum(
        actual == predicted
        for actual, predicted in zip(truth, prediction, strict=True)
    )
    accuracy = correct / max(1, len(truth))
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_scores) / max(1, len(f1_scores)),
        "per_class": per_class,
    }


def expected_calibration_error(
    *, confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> float:
    if len(confidences) != len(correct):
        raise ValueError("confidence and correctness lengths differ")
    if bins <= 0:
        raise ValueError("bins must be positive")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, is_correct in zip(confidences, correct, strict=True):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        buckets[min(int(confidence * bins), bins - 1)].append(
            (confidence, is_correct)
        )
    total = max(1, len(confidences))
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        error += len(bucket) / total * abs(accuracy - mean_confidence)
    return error


def false_positives_per_10k(false_positives: int, negative_examples: int) -> float:
    if false_positives < 0 or negative_examples < 0:
        raise ValueError("counts cannot be negative")
    return false_positives / max(1, negative_examples) * 10_000
