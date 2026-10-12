from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime

from pii_id.inference.mixed_scanner import CandidateScanResult
from pii_id.integration.dgx import (
    DgxEvidenceReference,
    DgxObservationContext,
    build_page_observations,
)
from pii_id.recognition.mixed_ids import IdentityCandidate


class DgxObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = DgxObservationContext(
            client_id="00000000-0000-4000-8000-000000000001",
            scan_run_id="00000000-0000-4000-8000-000000000002",
            work_item_id="00000000-0000-4000-8000-000000000003",
            generation_id="00000000-0000-4000-8000-000000000004",
            root_file_version_id="00000000-0000-4000-8000-000000000005",
            content_node_id="00000000-0000-4000-8000-000000000006",
            source_artifact_id="00000000-0000-4000-8000-000000000007",
            fragment_id="00000000-0000-4000-8000-000000000008",
            policy_digest="1" * 64,
            pipeline_digest="2" * 64,
            runtime_manifest_digest="3" * 64,
            detector_id="pii-id.mixed-identities-v2",
            detector_digest="4" * 64,
            model_digest="5" * 64,
            page=7,
            image_width=1000,
            image_height=800,
            evidence=DgxEvidenceReference(
                artifact_id="00000000-0000-4000-8000-000000000009",
                byte_offset=10,
                byte_length=100,
                sha256="6" * 64,
            ),
            created_at=datetime(2026, 10, 11, 12, 0, tzinfo=UTC),
        )

    @staticmethod
    def _candidate(*, value: str, x: int = 100) -> IdentityCandidate:
        return IdentityCandidate(
            id_type="IN_AADHAAR",
            country="IN",
            raw_value=value,
            polygon=((x, 200), (x + 200, 200), (x + 200, 240), (x, 240)),
            ocr_confidence=0.97,
            context_probability=0.98,
            format_valid=True,
            checksum_valid=True,
            source="ocr-pass-1",
            ocr_pass_count=3,
            visual_type_score=0.99,
            visual_presence_score=0.98,
            quality_codes=("BLUR",),
            validation_codes=("AADHAAR_VERHOEFF_VALID",),
        )

    def test_builds_raw_free_grounded_a3_observation(self) -> None:
        raw_value = "234567890125"
        scan = CandidateScanResult((self._candidate(value=raw_value),), ())

        observations = build_page_observations(
            scan, context=self.context, fingerprint_key=b"k" * 32
        )

        self.assertEqual(len(observations), 1)
        observation = observations[0]
        self.assertEqual(observation["contract"], "dgx.observation.v1")
        self.assertEqual(observation["category_id"], "identity.india")
        self.assertEqual(observation["source_locator"]["page"], 7)
        self.assertEqual(
            observation["source_locator"]["bounding_box"],
            [100.0, 200.0, 200.0, 40.0],
        )
        self.assertEqual(observation["candidate_validation"], "valid")
        self.assertEqual(observation["uncertainty"], "low")
        self.assertNotEqual(observation["masked_value"], raw_value)
        self.assertNotIn(raw_value, json.dumps(observations, sort_keys=True))
        self.assertNotIn("raw_value", observation)
        self.assertNotIn("raw_text", observation)
        self.assertNotIn("findings", observation)

    def test_duplicate_occurrences_have_distinct_stable_observation_ids(self) -> None:
        raw_value = "234567890125"
        scan = CandidateScanResult(
            (
                self._candidate(value=raw_value, x=100),
                self._candidate(value=raw_value, x=500),
            ),
            (),
        )

        first = build_page_observations(
            scan, context=self.context, fingerprint_key=b"k" * 32
        )
        second = build_page_observations(
            scan, context=self.context, fingerprint_key=b"k" * 32
        )

        self.assertNotEqual(first[0]["observation_id"], first[1]["observation_id"])
        self.assertEqual(
            [item["observation_id"] for item in first],
            [item["observation_id"] for item in second],
        )
        self.assertEqual(
            first[0]["identity_fingerprint"], first[1]["identity_fingerprint"]
        )

    def test_invalid_candidate_is_not_promoted_to_valid(self) -> None:
        candidate = self._candidate(value="234567890123")
        candidate = IdentityCandidate(
            **{
                **candidate.__dict__,
                "checksum_valid": False,
            }
        )

        observations = build_page_observations(
            CandidateScanResult((candidate,), ()),
            context=self.context,
            fingerprint_key=b"k" * 32,
        )

        self.assertEqual(observations[0]["candidate_validation"], "invalid")
        self.assertEqual(observations[0]["uncertainty"], "high")
        self.assertIn("CHECKSUM_INVALID", observations[0]["signal_codes"])

    def test_entity_specific_validation_signals_are_preserved_for_a4(self) -> None:
        observation = build_page_observations(
            CandidateScanResult((self._candidate(value="234567890125"),), ()),
            context=self.context,
            fingerprint_key=b"k" * 32,
        )[0]

        self.assertIn("AADHAAR_VERHOEFF_VALID", observation["signal_codes"])

    def test_rejects_weak_fingerprint_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 32 bytes"):
            build_page_observations(
                CandidateScanResult((self._candidate(value="234567890125"),), ()),
                context=self.context,
                fingerprint_key=b"too-short",
            )


if __name__ == "__main__":
    unittest.main()
