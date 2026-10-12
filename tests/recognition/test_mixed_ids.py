from __future__ import annotations

import unittest
from dataclasses import replace

from pii_id.recognition.mixed_ids import (
    extract_identity_candidates,
    merge_identity_passes,
    tokens_from_text,
)
from pii_id.recognition.validators import aadhaar_check_digit


class MixedIdentityCandidateTests(unittest.TestCase):
    def test_india_identity_types_are_discovered_in_one_mixed_document(self) -> None:
        base = "23456789012"
        aadhaar = base + aadhaar_check_digit(base)
        text = (
            f"Aadhaar Number {aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}\n"
            "Permanent Account Number PAN ABCPD1234F\n"
            "Election Commission Voter ID EPIC ABC1234567\n"
            "ABHA Number 12-3456-7890-1234\n"
            "Passport No A1234567\n"
            "Driving Licence DL-1420110012345"
        )

        candidates = extract_identity_candidates(tokens_from_text(text))

        self.assertEqual(
            {candidate.id_type for candidate in candidates},
            {
                "IN_AADHAAR",
                "IN_PAN",
                "IN_EPIC",
                "IN_ABHA",
                "IN_PASSPORT",
                "IN_DRIVING_LICENCE",
            },
        )
        aadhaar_candidate = next(
            candidate for candidate in candidates if candidate.id_type == "IN_AADHAAR"
        )
        self.assertTrue(aadhaar_candidate.checksum_valid)
        self.assertGreaterEqual(aadhaar_candidate.context_probability, 0.9)

    def test_priority_country_national_ids_require_country_context(self) -> None:
        text = (
            "Nepal National Identity Number 1234567890\n"
            "Bangladesh NID 1234567890123\n"
            "Sri Lanka NIC 855420159V"
        )

        candidates = extract_identity_candidates(tokens_from_text(text))

        self.assertEqual(
            {candidate.id_type for candidate in candidates},
            {"NP_NATIONAL_ID", "BD_NID", "LK_NIC"},
        )

    def test_multilingual_context_can_identify_pan_and_epic(self) -> None:
        candidates = extract_identity_candidates(
            tokens_from_text(
                "आयकर विभाग स्थायी खाता संख्या ABCPD1234F\n"
                "मतदाता पहचान पत्र ABC1234567"
            )
        )

        self.assertEqual(
            {candidate.id_type for candidate in candidates},
            {"IN_PAN", "IN_EPIC"},
        )
        self.assertTrue(
            all(candidate.context_probability >= 0.9 for candidate in candidates)
        )

    def test_driving_licence_accepts_official_slash_separated_display(self) -> None:
        candidates = extract_identity_candidates(
            tokens_from_text("Driving Licence No CH01 / 2013/0006875")
        )

        driving_licences = [
            candidate
            for candidate in candidates
            if candidate.id_type == "IN_DRIVING_LICENCE"
        ]
        self.assertEqual(
            [candidate.raw_value for candidate in driving_licences],
            ["CH0120130006875"],
        )

    def test_candidates_expose_entity_specific_validation_signals(self) -> None:
        candidates = extract_identity_candidates(
            tokens_from_text("Permanent Account Number PAN ABCPD1234F")
        )

        self.assertEqual(len(candidates), 1)
        self.assertIn("PAN_HOLDER_TYPE_VALID", candidates[0].validation_codes)

    def test_unlabelled_numbers_do_not_become_context_only_national_ids(self) -> None:
        candidates = extract_identity_candidates(
            tokens_from_text("Invoice 1234567890 total 1234567890123")
        )

        self.assertNotIn(
            "NP_NATIONAL_ID", {candidate.id_type for candidate in candidates}
        )
        self.assertNotIn("BD_NID", {candidate.id_type for candidate in candidates})

    def test_birth_date_digits_are_not_promoted_to_national_id_by_visual_context(
        self,
    ) -> None:
        candidates = extract_identity_candidates(
            tokens_from_text("DATE OF BIRTH\n4710511986", source="ocr-test"),
            contextless_entities=frozenset({"NP_NATIONAL_ID"}),
        )

        self.assertNotIn(
            "NP_NATIONAL_ID", {candidate.id_type for candidate in candidates}
        )

    def test_generic_identifier_label_allows_visually_supported_national_id(
        self,
    ) -> None:
        candidates = extract_identity_candidates(
            tokens_from_text("IDENTIFIER\n2060120631", source="ocr-test"),
            contextless_entities=frozenset({"NP_NATIONAL_ID"}),
        )

        self.assertIn(
            "NP_NATIONAL_ID", {candidate.id_type for candidate in candidates}
        )

    def test_pass_merge_counts_independent_agreement(self) -> None:
        text = "Permanent Account Number PAN ABCPD1234F"
        first = extract_identity_candidates(
            tokens_from_text(text, source="ocr-original")
        )
        second = extract_identity_candidates(tokens_from_text(text, source="ocr-gray"))

        merged = merge_identity_passes((first, second))

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].ocr_pass_count, 2)

    def test_repeated_value_is_preserved_as_separate_occurrences(self) -> None:
        text = (
            "Permanent Account Number PAN ABCPD1234F\n"
            "Duplicate copy PAN ABCPD1234F"
        )

        candidates = extract_identity_candidates(tokens_from_text(text))

        pan_candidates = [
            candidate for candidate in candidates if candidate.id_type == "IN_PAN"
        ]
        self.assertEqual(len(pan_candidates), 2)
        self.assertNotEqual(pan_candidates[0].polygon, pan_candidates[1].polygon)

    def test_pass_merge_fuses_views_without_collapsing_occurrences(self) -> None:
        text = (
            "Permanent Account Number PAN ABCPD1234F\n"
            "Duplicate copy PAN ABCPD1234F"
        )
        first = extract_identity_candidates(
            tokens_from_text(text, source="ocr-original")
        )
        second = extract_identity_candidates(tokens_from_text(text, source="ocr-gray"))

        merged = merge_identity_passes((first, second))

        pan_candidates = [
            candidate for candidate in merged if candidate.id_type == "IN_PAN"
        ]
        self.assertEqual(len(pan_candidates), 2)
        self.assertEqual(
            [candidate.ocr_pass_count for candidate in pan_candidates], [2, 2]
        )

    def test_context_gated_repairs_recover_position_specific_ocr_confusions(
        self,
    ) -> None:
        cases = {
            "Permanent Account Number IDENTIFIER FJIPS91211": (
                "IN_PAN",
                "FJIPS9121I",
            ),
            "Election Commission EPIC IDENTIFIER NX1I1792422": (
                "IN_EPIC",
                "NXI1792422",
            ),
            "Passport Number IDENTIFIER 22907801": (
                "IN_PASSPORT",
                "Z2907801",
            ),
            "Passport Number IDENTIFIER $4403423": (
                "IN_PASSPORT",
                "S4403423",
            ),
            "Passport Number IDENTIFIER Z 2907801": (
                "IN_PASSPORT",
                "Z2907801",
            ),
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                candidates = extract_identity_candidates(
                    tokens_from_text(text, source="ocr-test")
                )
                repaired = [
                    candidate
                    for candidate in candidates
                    if "OCR_POSITIONAL_REPAIR" in candidate.validation_codes
                ]
                self.assertIn(
                    expected,
                    {(item.id_type, item.raw_value) for item in repaired},
                )

    def test_ocr_repairs_require_document_and_field_context(self) -> None:
        candidates = extract_identity_candidates(
            tokens_from_text("Invoice 22907801", source="ocr-test")
        )

        self.assertNotIn(
            "IN_PASSPORT", {candidate.id_type for candidate in candidates}
        )

    def test_native_text_is_never_changed_by_ocr_confusion_repair(self) -> None:
        candidates = extract_identity_candidates(
            tokens_from_text("Passport Number IDENTIFIER 22907801")
        )

        self.assertNotIn(
            "IN_PASSPORT", {candidate.id_type for candidate in candidates}
        )

    def test_competing_ocr_values_for_one_occurrence_are_not_both_confirmable(
        self,
    ) -> None:
        correct_text = "Permanent Account Number PAN ABCPD1234F"
        wrong_text = "Permanent Account Number PAN ABCPD1234P"
        correct_first = extract_identity_candidates(
            tokens_from_text(correct_text, source="ocr-original")
        )
        correct_second = extract_identity_candidates(
            tokens_from_text(correct_text, source="ocr-gray")
        )
        correct_third = extract_identity_candidates(
            tokens_from_text(correct_text, source="ocr-sharp")
        )
        wrong_first = extract_identity_candidates(
            tokens_from_text(wrong_text, source="ocr-original")
        )
        wrong_second = extract_identity_candidates(
            tokens_from_text(wrong_text, source="ocr-gray")
        )

        merged = merge_identity_passes(
            (
                correct_first + wrong_first,
                correct_second + wrong_second,
                correct_third,
            )
        )

        correct = next(item for item in merged if item.raw_value == "ABCPD1234F")
        superseded = next(item for item in merged if item.raw_value == "ABCPD1234P")
        self.assertNotIn("OCR_VALUE_CONFLICT", correct.quality_codes)
        self.assertIn("OCR_VALUE_SUPERSEDED", superseded.quality_codes)

    def test_rectified_route_breaks_equal_vote_for_one_occurrence(self) -> None:
        correct = extract_identity_candidates(
            tokens_from_text(
                "Permanent Account Number PAN ABCPD1234F", source="ocr-rectified"
            )
        )[0]
        correct = replace(
            correct,
            validation_codes=(*correct.validation_codes, "RECTIFIED_OCR"),
        )
        wrong = extract_identity_candidates(
            tokens_from_text(
                "Permanent Account Number PAN ABCPD1234P", source="ocr-full"
            )
        )[0]

        merged = merge_identity_passes(((correct, wrong),))

        correct_merged = next(item for item in merged if item.raw_value.endswith("F"))
        wrong_merged = next(item for item in merged if item.raw_value.endswith("P"))
        self.assertNotIn("OCR_VALUE_CONFLICT", correct_merged.quality_codes)
        self.assertIn("OCR_VALUE_SUPERSEDED", wrong_merged.quality_codes)


if __name__ == "__main__":
    unittest.main()
