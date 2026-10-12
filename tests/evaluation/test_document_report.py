from __future__ import annotations

import unittest

from pii_id.evaluation.document_report import summarize_document_predictions


class DocumentReportTests(unittest.TestCase):
    def test_summary_exposes_strict_detection_and_slice_failures(self) -> None:
        rows = [
            {
                "truth_type": "A",
                "predicted_type": "A",
                "truth_presence": True,
                "predicted_presence": True,
                "polygon_iou": 0.8,
                "country": "IN",
                "script": "devanagari",
                "modality": "template",
                "dataset_id": "one",
            },
            {
                "truth_type": "B",
                "predicted_type": "A",
                "truth_presence": True,
                "predicted_presence": False,
                "polygon_iou": 0.4,
                "country": "NP",
                "script": "devanagari",
                "modality": "camera_capture",
                "dataset_id": "two",
            },
        ]

        report = summarize_document_predictions(rows)

        self.assertEqual(report["overall"]["examples"], 2)
        self.assertEqual(report["overall"]["recall_iou_50"], 0.5)
        self.assertEqual(report["overall"]["recall_iou_75"], 0.5)
        self.assertEqual(report["overall"]["document_type_accuracy"], 0.5)
        self.assertEqual(report["overall"]["presence_accuracy"], 0.5)
        self.assertEqual(report["overall"]["false_negative_rate"], 0.5)
        self.assertEqual(report["by_country"]["NP"]["mean_polygon_iou"], 0.4)
        self.assertIn("B", report["classification"]["per_class"])

    def test_non_id_rows_are_excluded_from_type_and_geometry_metrics(self) -> None:
        rows = [
            {
                "truth_type": "A",
                "predicted_type": "A",
                "truth_presence": True,
                "predicted_presence": True,
                "polygon_iou": 0.9,
                "country": "IN",
                "script": "latin",
                "modality": "template",
                "dataset_id": "one",
            },
            {
                "truth_type": "NON_ID",
                "predicted_type": "A",
                "truth_presence": False,
                "predicted_presence": True,
                "polygon_iou": 0.0,
                "country": "IN",
                "script": "latin",
                "modality": "hospital_form",
                "dataset_id": "synthetic",
            },
        ]

        report = summarize_document_predictions(rows)

        self.assertEqual(report["overall"]["mean_polygon_iou"], 0.9)
        self.assertEqual(report["overall"]["document_type_accuracy"], 1.0)
        self.assertEqual(report["overall"]["false_positive_rate"], 1.0)
        self.assertEqual(report["overall"]["positive_examples"], 1)
        self.assertEqual(report["overall"]["negative_examples"], 1)


if __name__ == "__main__":
    unittest.main()
