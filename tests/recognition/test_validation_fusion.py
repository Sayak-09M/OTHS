from __future__ import annotations

import unittest

from pii_id.recognition.fusion import (
    CandidateEvidence,
    ConfirmationPolicy,
    fit_precision_constrained_threshold,
    fuse_identity_candidate,
)
from pii_id.recognition.validators import (
    aadhaar_check_digit,
    mask_identifier,
    normalize_decimal_digits,
    validate_aadhaar,
    validate_pan_syntax,
)


class IdentityValidatorTests(unittest.TestCase):
    def test_aadhaar_verhoeff_accepts_generated_check_digit_and_native_digits(
        self,
    ) -> None:
        base = "23456789012"
        candidate = base + aadhaar_check_digit(base)

        self.assertTrue(validate_aadhaar(candidate))
        invalid = candidate[:-1] + str((int(candidate[-1]) + 1) % 10)
        self.assertFalse(validate_aadhaar(invalid))
        self.assertEqual(normalize_decimal_digits("१२३ ৪৫"), "12345")

    def test_pan_syntax_is_validation_evidence_not_confirmation(self) -> None:
        self.assertTrue(validate_pan_syntax("ABCDE1234F"))
        self.assertFalse(validate_pan_syntax("ABC1E1234F"))
        self.assertEqual(mask_identifier("ABCDE1234F"), "*****1234F")


class FusionPolicyTests(unittest.TestCase):
    def test_calibration_selects_lowest_threshold_meeting_precision_target(
        self,
    ) -> None:
        fitted = fit_precision_constrained_threshold(
            scores=[0.99, 0.95, 0.90, 0.80, 0.70],
            labels=[True, True, False, True, False],
            target_precision=0.75,
        )

        self.assertTrue(fitted.enabled)
        self.assertEqual(fitted.threshold, 0.80)
        self.assertEqual(fitted.precision, 0.75)
        self.assertEqual(fitted.recall, 1.0)

    def test_aadhaar_requires_visual_context_and_checksum_before_confirmation(
        self,
    ) -> None:
        base = "23456789012"
        valid_number = base + aadhaar_check_digit(base)
        policy = ConfirmationPolicy(
            id_type="IN_AADHAAR",
            clean_threshold=0.90,
            degraded_threshold=0.97,
            enabled=True,
        )
        strong = CandidateEvidence(
            id_type="IN_AADHAAR",
            raw_value=valid_number,
            polygon=((1, 2), (10, 2), (10, 5), (1, 5)),
            visual_probability=0.98,
            keyword_probability=0.95,
            ocr_confidence=0.94,
            quality_bucket="degraded",
            checksum_valid=True,
            cryptographic_valid=False,
            ocr_pass_count=2,
        )

        finding = fuse_identity_candidate(strong, policy)
        no_checksum = fuse_identity_candidate(
            CandidateEvidence(**{**strong.__dict__, "checksum_valid": False}), policy
        )
        no_context = fuse_identity_candidate(
            CandidateEvidence(**{**strong.__dict__, "keyword_probability": 0.1}),
            policy,
        )

        self.assertIsNotNone(finding)
        self.assertEqual(finding.status, "confirmed")
        self.assertNotIn(valid_number, finding.masked_value)
        self.assertIsNone(no_checksum)
        self.assertIsNone(no_context)

    def test_disabled_id_type_never_emits_confirmed_finding(self) -> None:
        policy = ConfirmationPolicy(
            id_type="IN_PAN",
            clean_threshold=0.99,
            degraded_threshold=0.995,
            enabled=False,
        )
        evidence = CandidateEvidence(
            id_type="IN_PAN",
            raw_value="ABCDE1234F",
            polygon=((0, 0), (1, 0), (1, 1), (0, 1)),
            visual_probability=1.0,
            keyword_probability=1.0,
            ocr_confidence=1.0,
            quality_bucket="clean",
            checksum_valid=True,
            cryptographic_valid=False,
            ocr_pass_count=3,
        )

        self.assertIsNone(fuse_identity_candidate(evidence, policy))


if __name__ == "__main__":
    unittest.main()
