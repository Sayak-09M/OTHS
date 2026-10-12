from __future__ import annotations

import unittest

import torch
from PIL import Image

from pii_id.inference.mixed_scanner import (
    IdentityPolicy,
    MixedIdentityScanner,
)
from pii_id.recognition.mixed_ids import extract_identity_candidates, tokens_from_text
from pii_id.recognition.validators import aadhaar_check_digit


class FakeOcr:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def recognize(self, image: Image.Image, *, scripts: tuple[str, ...]):
        self.calls += 1
        return tokens_from_text(self.text, source=f"ocr-pass-{self.calls}")


class FakeDocumentModel(torch.nn.Module):
    def __init__(
        self,
        *,
        presence_logit: float = 8.0,
        type_logits: tuple[float, ...] = (-5.0, 8.0),
    ) -> None:
        super().__init__()
        self.presence_logit = presence_logit
        self.type_logits = type_logits

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        return {
            "presence_logits": torch.full((batch,), self.presence_logit),
            "corners": torch.tensor(
                [[[0.0, 0.25], [1.0, 0.25], [1.0, 0.75], [0.0, 0.75]]]
            ).repeat(batch, 1, 1),
            "type_logits": torch.tensor([self.type_logits]).repeat(batch, 1),
            "quality_logits": torch.tensor(
                [[8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0]]
            ).repeat(batch, 1),
        }


class BrokenDocumentModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        raise RuntimeError("synthetic model failure")


class RectifiedOnlyOcr:
    def __init__(self, text: str, rectified_size: tuple[int, int]) -> None:
        self.text = text
        self.rectified_size = rectified_size
        self.calls: list[tuple[int, int]] = []

    def recognize(self, image: Image.Image, *, scripts: tuple[str, ...]):
        self.calls.append(image.size)
        text = self.text if image.size == self.rectified_size else ""
        return tokens_from_text(text, source=f"ocr-pass-{len(self.calls)}")


class CompetingValueOcr:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image: Image.Image, *, scripts: tuple[str, ...]):
        self.calls += 1
        value = "ABCPD1234P" if self.calls in {2, 5} else "ABCPD1234F"
        return tokens_from_text(
            f"Permanent Account Number PAN {value}",
            source=f"ocr-pass-{self.calls}",
        )


class MixedIdentityScannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policies = {
            "IN_AADHAAR": IdentityPolicy(
                id_type="IN_AADHAAR",
                enabled=True,
                require_checksum=True,
                minimum_context_probability=0.8,
                minimum_ocr_confidence=0.7,
                minimum_ocr_passes=2,
            ),
            "IN_PAN": IdentityPolicy(
                id_type="IN_PAN",
                enabled=True,
                require_checksum=False,
                minimum_context_probability=0.8,
                minimum_ocr_confidence=0.7,
                minimum_ocr_passes=2,
            ),
        }

    def test_image_ocr_discovers_types_without_visual_classification_gate(self) -> None:
        base = "23456789012"
        aadhaar = base + aadhaar_check_digit(base)
        ocr = FakeOcr(
            f"Aadhaar {aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}\n"
            "Permanent Account Number PAN ABCPD1234F"
        )
        scanner = MixedIdentityScanner(ocr=ocr, policies=self.policies)

        result = scanner.scan_image(Image.new("RGB", (300, 200), "white"))

        self.assertEqual(ocr.calls, 3)
        self.assertEqual(
            {finding.id_type for finding in result.findings},
            {"IN_AADHAAR", "IN_PAN"},
        )
        self.assertEqual(result.review_candidates, ())

    def test_native_text_is_reliable_without_repeated_ocr_passes(self) -> None:
        scanner = MixedIdentityScanner(ocr=FakeOcr(""), policies=self.policies)

        result = scanner.scan_text("Permanent Account Number PAN ABCPD1234F")

        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].id_type, "IN_PAN")
        self.assertIn("native_text", result.findings[0].evidence)

    def test_visual_document_evidence_can_supply_missing_text_context(self) -> None:
        scanner = MixedIdentityScanner(
            ocr=FakeOcr("ABCPD1234F"),
            policies=self.policies,
            model=FakeDocumentModel(type_logits=(-8.0, -8.0, 8.0)),
            label_classes=("UNKNOWN", "IN_AADHAAR", "IN_PAN"),
            device=torch.device("cpu"),
        )

        result = scanner.scan_image(Image.new("RGB", (300, 200), "white"))

        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].id_type, "IN_PAN")
        self.assertIn("context_source=visual", result.findings[0].evidence)

    def test_visual_document_evidence_enables_context_gated_passport_discovery(
        self,
    ) -> None:
        passport_policy = IdentityPolicy(
            id_type="IN_PASSPORT",
            enabled=True,
            require_checksum=False,
            minimum_context_probability=0.8,
            minimum_ocr_confidence=0.7,
            minimum_ocr_passes=2,
        )
        scanner = MixedIdentityScanner(
            ocr=FakeOcr("A1234567"),
            policies={"IN_PASSPORT": passport_policy},
            model=FakeDocumentModel(type_logits=(-8.0, 8.0)),
            label_classes=("UNKNOWN", "IN_PASSPORT"),
            device=torch.device("cpu"),
        )

        result = scanner.scan_image(Image.new("RGB", (300, 200), "white"))

        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].id_type, "IN_PASSPORT")
        self.assertIn("context_source=visual", result.findings[0].evidence)

    def test_rejected_candidate_is_reported_for_review_with_reasons(self) -> None:
        scanner = MixedIdentityScanner(
            ocr=FakeOcr("Aadhaar 2345 6789 0123"), policies=self.policies
        )

        result = scanner.scan_image(Image.new("RGB", (300, 200), "white"))

        self.assertEqual(result.findings, ())
        self.assertEqual(len(result.review_candidates), 1)
        self.assertIn("checksum_invalid", result.review_candidates[0].rejection_reasons)

    def test_native_text_reports_every_repeated_id_occurrence(self) -> None:
        scanner = MixedIdentityScanner(ocr=FakeOcr(""), policies=self.policies)

        result = scanner.scan_text(
            "Permanent Account Number PAN ABCPD1234F\n"
            "Duplicate copy PAN ABCPD1234F"
        )

        self.assertEqual(len(result.findings), 2)
        self.assertTrue(all(item.id_type == "IN_PAN" for item in result.findings))
        self.assertNotEqual(result.findings[0].polygon, result.findings[1].polygon)

    def test_model_rectification_adds_ocr_without_gating_full_page(self) -> None:
        base = "23456789012"
        aadhaar = base + aadhaar_check_digit(base)
        rectified_size = (640, 400)
        ocr = RectifiedOnlyOcr(
            f"Aadhaar {aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}",
            rectified_size,
        )
        scanner = MixedIdentityScanner(
            ocr=ocr,
            policies=self.policies,
            model=FakeDocumentModel(),
            label_classes=("UNKNOWN", "IN_AADHAAR"),
            device=torch.device("cpu"),
            rectified_size=rectified_size,
        )

        result = scanner.scan_image(Image.new("RGB", (200, 100), "white"))

        self.assertEqual(ocr.calls.count((200, 100)), 3)
        self.assertEqual(ocr.calls.count(rectified_size), 3)
        self.assertEqual(len(result.findings), 1)
        self.assertIn(
            "visual_type_score=0.999998", result.findings[0].evidence
        )
        self.assertIn("quality=BLUR", result.findings[0].evidence)
        self.assertTrue(
            all(
                0 <= coordinate <= limit
                for point in result.findings[0].polygon
                for coordinate, limit in zip(point, (200, 100), strict=True)
            )
        )

    def test_low_visual_presence_never_suppresses_full_page_ocr(self) -> None:
        base = "23456789012"
        aadhaar = base + aadhaar_check_digit(base)
        ocr = FakeOcr(f"Aadhaar {aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}")
        scanner = MixedIdentityScanner(
            ocr=ocr,
            policies=self.policies,
            model=FakeDocumentModel(presence_logit=-8.0),
            label_classes=("UNKNOWN", "IN_AADHAAR"),
            device=torch.device("cpu"),
        )

        result = scanner.scan_image(Image.new("RGB", (300, 200), "white"))

        self.assertEqual(ocr.calls, 3)
        self.assertEqual(len(result.findings), 1)
        self.assertIn("visual_presence_score=0.000335", result.findings[0].evidence)

    def test_visual_failure_never_suppresses_full_page_ocr(self) -> None:
        base = "23456789012"
        aadhaar = base + aadhaar_check_digit(base)
        ocr = FakeOcr(f"Aadhaar {aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}")
        scanner = MixedIdentityScanner(
            ocr=ocr,
            policies=self.policies,
            model=BrokenDocumentModel(),
            label_classes=("UNKNOWN", "IN_AADHAAR"),
            device=torch.device("cpu"),
        )

        result = scanner.scan_image(Image.new("RGB", (300, 200), "white"))

        self.assertEqual(ocr.calls, 3)
        self.assertEqual(len(result.findings), 1)
        self.assertIn(
            "visual_inference_failed", {event.code for event in result.events}
        )

    def test_candidate_scan_retains_raw_value_only_in_memory_for_a3_adapter(
        self,
    ) -> None:
        base = "23456789012"
        aadhaar = base + aadhaar_check_digit(base)
        scanner = MixedIdentityScanner(
            ocr=FakeOcr(f"Aadhaar {aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}"),
            policies=self.policies,
        )

        result = scanner.scan_image_candidates(
            Image.new("RGB", (300, 200), "white")
        )

        self.assertEqual(result.candidates[0].raw_value, aadhaar)
        self.assertEqual(result.candidates[0].ocr_pass_count, 3)

    def test_candidate_rejection_reasons_use_the_loaded_policy(self) -> None:
        scanner = MixedIdentityScanner(
            ocr=FakeOcr("Aadhaar 2345 6789 0123"), policies=self.policies
        )
        candidate = scanner.scan_image_candidates(
            Image.new("RGB", (300, 200), "white")
        ).candidates[0]

        reasons = scanner.candidate_rejection_reasons(candidate)

        self.assertIn("checksum_invalid", reasons)

    def test_reference_validation_excludes_invalid_aadhaar_ground_truth(self) -> None:
        scanner = MixedIdentityScanner(ocr=FakeOcr(""), policies=self.policies)
        base = "23456789012"
        valid = base + aadhaar_check_digit(base)

        self.assertEqual(
            scanner.reference_value_rejection_reasons("IN_AADHAAR", valid), ()
        )
        self.assertIn(
            "checksum_invalid",
            scanner.reference_value_rejection_reasons(
                "IN_AADHAAR", "234567890123"
            ),
        )

    def test_strong_multi_pass_consensus_can_use_a_lower_confidence_floor(self) -> None:
        policy = IdentityPolicy(
            id_type="IN_PAN",
            enabled=True,
            require_checksum=False,
            minimum_context_probability=0.8,
            minimum_ocr_confidence=0.75,
            minimum_ocr_passes=2,
            strong_consensus_ocr_passes=3,
            strong_consensus_minimum_ocr_confidence=0.60,
        )
        scanner = MixedIdentityScanner(ocr=FakeOcr(""), policies={"IN_PAN": policy})
        candidate = extract_identity_candidates(
            tokens_from_text("Permanent Account Number PAN ABCPD1234F")
        )[0]
        candidate = candidate.__class__(
            **{
                **candidate.__dict__,
                "ocr_confidence": 0.65,
                "ocr_pass_count": 3,
            }
        )

        self.assertEqual(scanner.candidate_rejection_reasons(candidate), ())

    def test_superseded_competing_ocr_value_is_never_confirmed(self) -> None:
        scanner = MixedIdentityScanner(ocr=FakeOcr(""), policies=self.policies)
        candidate = extract_identity_candidates(
            tokens_from_text("Permanent Account Number PAN ABCPD1234F")
        )[0]
        candidate = candidate.__class__(
            **{
                **candidate.__dict__,
                "quality_codes": ("OCR_VALUE_SUPERSEDED",),
                "ocr_pass_count": 3,
            }
        )

        self.assertIn(
            "competing_ocr_value", scanner.candidate_rejection_reasons(candidate)
        )

    def test_visual_enrichment_preserves_competing_value_suppression(self) -> None:
        scanner = MixedIdentityScanner(
            ocr=CompetingValueOcr(),
            policies=self.policies,
            model=FakeDocumentModel(type_logits=(-8.0, -8.0, 8.0)),
            label_classes=("UNKNOWN", "IN_AADHAAR", "IN_PAN"),
            device=torch.device("cpu"),
        )

        result = scanner.scan_image(Image.new("RGB", (300, 200), "white"))

        self.assertEqual(
            [finding.masked_value for finding in result.findings], ["*****1234F"]
        )
        self.assertEqual(len(result.review_candidates), 1)
        self.assertIn(
            "competing_ocr_value",
            result.review_candidates[0].rejection_reasons,
        )


if __name__ == "__main__":
    unittest.main()
