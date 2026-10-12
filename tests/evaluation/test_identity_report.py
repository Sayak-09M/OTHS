from __future__ import annotations

import json
import unittest

from pii_id.evaluation.identity_report import (
    evaluate_identity_record,
    summarize_identity_evaluation,
)
from pii_id.recognition.mixed_ids import IdentityCandidate


def _record(
    *,
    record_id: str,
    document_type: str,
    value: str | None,
    modality: str = "camera_degraded",
) -> dict[str, object]:
    fields = []
    if value is not None:
        fields.append(
            {
                "source_name": "number",
                "canonical_name": "id_number",
                "pii_type": f"{document_type}_NUMBER",
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "script": "latin",
                "value": value,
            }
        )
    return {
        "record_id": record_id,
        "dataset_id": "safe-fixture",
        "identity_group": f"group-{record_id}",
        "split": "validation",
        "country": "IN",
        "document_type": document_type,
        "script": "latin",
        "modality": modality,
        "image": {
            "storage": "file",
            "path": f"fixtures/{record_id}.png",
            "width": 10,
            "height": 10,
            "size_bytes": 10,
        },
        "annotations": {
            "document_polygon": [[0, 0], [9, 0], [9, 9], [0, 9]],
            "document_out_of_frame": False,
            "fields": fields,
        },
        "provenance": {"content_class": "synthetic", "licence": "test"},
    }


def _candidate(id_type: str, value: str) -> IdentityCandidate:
    return IdentityCandidate(
        id_type=id_type,
        country="IN",
        raw_value=value,
        polygon=((0, 0), (1, 0), (1, 1), (0, 1)),
        ocr_confidence=0.9,
        context_probability=0.98,
        format_valid=True,
        checksum_valid=True if id_type == "IN_AADHAAR" else None,
        source="fixture",
        ocr_pass_count=2,
    )


class IdentityReportTests(unittest.TestCase):
    def test_report_separates_discovery_and_confirmation_and_masks_values(
        self,
    ) -> None:
        aadhaar = "234567890123"
        row = evaluate_identity_record(
            _record(
                record_id="aadhaar-rejected-234567890123",
                document_type="IN_AADHAAR",
                value="2345 6789 0123",
            ),
            candidates=(_candidate("IN_AADHAAR", aadhaar),),
            selected_entities={"IN_AADHAAR", "IN_PAN"},
            rejection_reasons=lambda candidate: (
                "ocr_pass_agreement_below_threshold",
            ),
            manifest_line=17,
        )

        report = summarize_identity_evaluation([row])

        self.assertEqual(report["candidate_level"]["recall"], 1.0)
        self.assertEqual(report["confirmed_level"]["recall"], 0.0)
        self.assertEqual(
            report["failures"][0]["truth_status"][0]["status"],
            "rejected_by_policy",
        )
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn(aadhaar, serialized)
        self.assertIn("********0123", serialized)
        self.assertEqual(report["failures"][0]["manifest_line"], 17)

    def test_unlabelled_supported_document_is_weak_coverage_not_negative(
        self,
    ) -> None:
        row = evaluate_identity_record(
            _record(
                record_id="unlabelled-aadhaar",
                document_type="IN_AADHAAR",
                value=None,
            ),
            candidates=(),
            selected_entities={"IN_AADHAAR", "IN_PAN"},
            rejection_reasons=lambda candidate: (),
        )

        report = summarize_identity_evaluation([row])

        self.assertEqual(report["coverage"]["exact_records"], 0)
        self.assertEqual(report["coverage"]["weak_positive_records"], 1)
        self.assertEqual(report["confirmed_level"]["true_negative_records"], 0)
        self.assertEqual(
            report["weak_positive_detection"]["IN_AADHAAR"]["confirmed_rate"],
            0.0,
        )

    def test_non_id_negative_deduplicates_candidates(self) -> None:
        pan = "ABCPD1234F"
        row = evaluate_identity_record(
            _record(record_id="non-id", document_type="NON_ID", value=None),
            candidates=(_candidate("IN_PAN", pan), _candidate("IN_PAN", pan)),
            selected_entities={"IN_AADHAAR", "IN_PAN"},
            rejection_reasons=lambda candidate: (),
        )

        report = summarize_identity_evaluation([row])

        self.assertEqual(report["confirmed_level"]["false_positives"], 1)
        self.assertEqual(report["confirmed_level"]["false_positive_records"], 1)
        self.assertEqual(report["confirmed_level"]["negative_records"], 1)

    def test_report_exposes_entity_and_modality_failures(self) -> None:
        correct_pan = evaluate_identity_record(
            _record(
                record_id="pan-pass",
                document_type="IN_PAN",
                value="abcpd 1234 f",
                modality="flat_scan",
            ),
            candidates=(_candidate("IN_PAN", "ABCPD1234F"),),
            selected_entities={"IN_AADHAAR", "IN_PAN"},
            rejection_reasons=lambda candidate: (),
        )
        missed_pan = evaluate_identity_record(
            _record(
                record_id="pan-miss",
                document_type="IN_PAN",
                value="ZZZZP9999Z",
                modality="camera_degraded",
            ),
            candidates=(),
            selected_entities={"IN_AADHAAR", "IN_PAN"},
            rejection_reasons=lambda candidate: (),
        )

        report = summarize_identity_evaluation([correct_pan, missed_pan])

        self.assertEqual(report["confirmed_level"]["recall"], 0.5)
        self.assertEqual(report["by_entity"]["IN_PAN"]["recall"], 0.5)
        self.assertEqual(
            report["by_modality"]["camera_degraded"]["false_negatives"], 1
        )

    def test_invalid_truth_scores_ocr_but_not_policy_confirmation(self) -> None:
        aadhaar = "234567890123"
        row = evaluate_identity_record(
            _record(
                record_id="invalid-aadhaar-ground-truth",
                document_type="IN_AADHAAR",
                value=aadhaar,
            ),
            candidates=(_candidate("IN_AADHAAR", aadhaar),),
            selected_entities={"IN_AADHAAR"},
            rejection_reasons=lambda candidate: ("checksum_invalid",),
            truth_rejection_reasons=lambda id_type, value: (
                "checksum_invalid",
            ),
            manifest_line=31,
        )

        report = summarize_identity_evaluation([row])

        self.assertEqual(report["candidate_level"]["recall"], 1.0)
        self.assertEqual(report["confirmed_level"]["false_negatives"], 0)
        self.assertEqual(report["coverage"]["invalid_truth_identifiers"], 1)
        self.assertEqual(
            report["ground_truth_issues"][0]["validation_reasons"],
            ("checksum_invalid",),
        )


if __name__ == "__main__":
    unittest.main()
