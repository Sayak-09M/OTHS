from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pii_id.integration.readiness import assess_readiness

ROOT = Path(__file__).resolve().parents[2]


class DetectionGodXReadinessTests(unittest.TestCase):
    def test_current_bundle_is_candidate_ready_but_not_production_qualified(
        self,
    ) -> None:
        report = assess_readiness(root=ROOT)

        self.assertEqual(report["status"], "candidate_ready")
        self.assertFalse(report["production_qualified"])
        self.assertEqual(report["producer_role"], "A3")
        self.assertEqual(report["output_contract"], "dgx.observation.v1")
        self.assertEqual(report["missing_visual_labels"], [])
        self.assertEqual(
            report["contract_digests"]["dgx.observation.v1"],
            "f9a7c16517b731e9ce3d02008d4cfb62bcba778af3f49272db281e7aff67617e",
        )

    def test_validation_report_must_match_checkpoint_digest(self) -> None:
        original = json.loads(
            (ROOT / "artifacts/reports/foundation-v2-validation.json").read_text(
                encoding="utf-8"
            )
        )
        original["checkpoint_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            report_path = Path(directory) / "mismatch.json"
            report_path.write_text(json.dumps(original), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "checkpoint digest"):
                assess_readiness(root=ROOT, validation_report_path=report_path)


if __name__ == "__main__":
    unittest.main()
