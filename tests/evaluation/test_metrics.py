from __future__ import annotations

import unittest

from pii_id.evaluation.metrics import (
    character_error_rate,
    classification_metrics,
    expected_calibration_error,
    false_positives_per_10k,
    polygon_iou,
    word_error_rate,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_polygon_iou_uses_quadrilateral_overlap(self) -> None:
        truth = [[0, 0], [10, 0], [10, 10], [0, 10]]
        prediction = [[5, 5], [15, 5], [15, 15], [5, 15]]

        self.assertAlmostEqual(polygon_iou(truth, prediction), 1 / 7)
        self.assertEqual(polygon_iou(truth, truth), 1.0)

    def test_ocr_metrics_match_hand_calculated_edits(self) -> None:
        self.assertAlmostEqual(character_error_rate([("abc", "adc")]), 1 / 3)
        self.assertAlmostEqual(
            word_error_rate([("one two three", "one four three")]), 1 / 3
        )
        self.assertEqual(character_error_rate([("", "")]), 0.0)

    def test_classification_metrics_report_each_class_and_macro_f1(self) -> None:
        report = classification_metrics(
            truth=["A", "A", "B", "B"],
            prediction=["A", "B", "B", "B"],
        )

        self.assertAlmostEqual(report["per_class"]["A"]["precision"], 1.0)
        self.assertAlmostEqual(report["per_class"]["A"]["recall"], 0.5)
        self.assertAlmostEqual(report["per_class"]["B"]["precision"], 2 / 3)
        self.assertAlmostEqual(report["macro_f1"], (2 / 3 + 0.8) / 2)

    def test_calibration_and_false_positive_rates_are_explicit(self) -> None:
        ece = expected_calibration_error(
            confidences=[0.9, 0.8, 0.4, 0.3],
            correct=[True, True, False, True],
            bins=2,
        )

        self.assertAlmostEqual(ece, 0.15)
        self.assertEqual(false_positives_per_10k(2, 4), 5000.0)


if __name__ == "__main__":
    unittest.main()
