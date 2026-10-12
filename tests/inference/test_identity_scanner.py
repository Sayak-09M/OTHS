from __future__ import annotations

import json
import unittest

import torch
from PIL import Image

from pii_id.inference.identity_scanner import IdentityDocumentScanner
from pii_id.recognition.fusion import ConfirmationPolicy
from pii_id.recognition.ocr import OcrToken
from pii_id.recognition.validators import aadhaar_check_digit


class FakeDocumentModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        return {
            "presence_logits": torch.full((batch,), 8.0),
            "corners": torch.tensor(
                [[[0.0, 0.25], [1.0, 0.25], [1.0, 0.75], [0.0, 0.75]]]
            ).repeat(batch, 1, 1),
            "type_logits": torch.tensor([[-5.0, 8.0]]).repeat(batch, 1),
            "quality_logits": torch.full((batch, 7), -8.0),
        }


class FakeOcr:
    def __init__(self, value: str) -> None:
        self.value = value
        self.calls = 0

    def recognize(self, image: Image.Image, *, scripts: tuple[str, ...]):
        self.calls += 1
        pieces = ("Aadhaar", self.value[:4], self.value[4:8], self.value[8:])
        return tuple(
            OcrToken(
                text=text,
                confidence=0.96,
                polygon=(
                    (index * 100, 20),
                    (index * 100 + 80, 20),
                    (index * 100 + 80, 50),
                    (index * 100, 50),
                ),
                line_id=(1, 1, 1, 1),
            )
            for index, text in enumerate(pieces)
        )


class IdentityDocumentScannerTests(unittest.TestCase):
    def test_confirmed_result_has_source_location_evidence_and_masked_value(
        self,
    ) -> None:
        base = "23456789012"
        value = base + aadhaar_check_digit(base)
        ocr = FakeOcr(value)
        scanner = IdentityDocumentScanner(
            model=FakeDocumentModel(),
            label_classes=("UNKNOWN", "IN_AADHAAR"),
            ocr=ocr,
            policies={
                "IN_AADHAAR": ConfirmationPolicy(
                    id_type="IN_AADHAAR",
                    clean_threshold=0.95,
                    degraded_threshold=0.98,
                    enabled=True,
                )
            },
            device=torch.device("cpu"),
            input_size=224,
            rectified_size=(640, 400),
        )

        result = scanner.scan_image(Image.new("RGB", (200, 100), color="white"))

        self.assertEqual(ocr.calls, 3)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].id_type, "IN_AADHAAR")
        self.assertNotEqual(result.findings[0].masked_value, value)
        self.assertEqual(len(result.findings[0].source_polygon), 4)
        serialized_logs = json.dumps([event.as_log() for event in result.events])
        self.assertNotIn(value, serialized_logs)
        self.assertIn("confirmed_finding", serialized_logs)

    def test_invalid_checksum_produces_no_finding(self) -> None:
        ocr = FakeOcr("234567890123")
        scanner = IdentityDocumentScanner(
            model=FakeDocumentModel(),
            label_classes=("UNKNOWN", "IN_AADHAAR"),
            ocr=ocr,
            policies={
                "IN_AADHAAR": ConfirmationPolicy(
                    id_type="IN_AADHAAR",
                    clean_threshold=0.95,
                    degraded_threshold=0.98,
                    enabled=True,
                )
            },
            device=torch.device("cpu"),
        )

        result = scanner.scan_image(Image.new("RGB", (200, 100), color="white"))

        self.assertEqual(result.findings, ())
        self.assertIn("no_confirmed_candidate", {event.code for event in result.events})
        rejection_event = next(
            event for event in result.events if event.code == "candidate_rejections"
        )
        self.assertEqual(rejection_event.counts["checksum_invalid"], 1)


if __name__ == "__main__":
    unittest.main()
