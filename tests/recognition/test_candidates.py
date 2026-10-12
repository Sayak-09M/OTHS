from __future__ import annotations

import unittest

from pii_id.recognition.candidates import (
    extract_aadhaar_candidates,
    merge_aadhaar_passes,
)
from pii_id.recognition.ocr import OcrToken
from pii_id.recognition.validators import aadhaar_check_digit


def token(text: str, left: int, *, confidence: float = 0.95) -> OcrToken:
    return OcrToken(
        text=text,
        confidence=confidence,
        polygon=((left, 10), (left + 35, 10), (left + 35, 24), (left, 24)),
        line_id=(1, 1, 1, 1),
    )


class AadhaarCandidateTests(unittest.TestCase):
    def test_extracts_split_number_with_location_checksum_and_context(self) -> None:
        base = "23456789012"
        value = base + aadhaar_check_digit(base)
        tokens = (
            token("Aadhaar", 0),
            token(value[:4], 60),
            token(value[4:8], 100),
            token(value[8:], 140),
        )

        candidates = extract_aadhaar_candidates(tokens)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].raw_value, value)
        self.assertTrue(candidates[0].checksum_valid)
        self.assertGreater(candidates[0].keyword_probability, 0.9)
        self.assertEqual(candidates[0].polygon[0], (60, 10))
        self.assertEqual(candidates[0].polygon[2], (175, 24))

    def test_two_ocr_passes_are_preserved_as_independent_agreement(self) -> None:
        base = "23456789012"
        value = base + aadhaar_check_digit(base)
        first = extract_aadhaar_candidates(
            (token("UIDAI", 0), token(value, 50, confidence=0.80))
        )
        second = extract_aadhaar_candidates(
            (token("Aadhaar", 0), token(value, 50, confidence=0.96))
        )

        merged = merge_aadhaar_passes((first, second))

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].ocr_pass_count, 2)
        self.assertEqual(merged[0].ocr_confidence, 0.96)


if __name__ == "__main__":
    unittest.main()
