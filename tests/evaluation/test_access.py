from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pii_id.evaluation.access import (
    AccessPurpose,
    FrozenCandidate,
    LockedTestAccessError,
    SplitAccessGuard,
)


class SplitAccessGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.ledger = self.root / "audit" / "dataset-access.jsonl"
        self.guard = SplitAccessGuard(root=self.root, ledger_path=self.ledger)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def ledger_events(self) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in self.ledger.read_text(encoding="utf-8").splitlines()
        ]

    def test_fit_cannot_access_locked_test_and_denial_is_logged(self) -> None:
        with self.assertRaises(LockedTestAccessError):
            self.guard.authorize(
                split="locked_test",
                purpose=AccessPurpose.FIT,
                run_id="run-001",
            )

        event = self.ledger_events()[0]
        self.assertEqual(event["decision"], "denied")
        self.assertEqual(event["split"], "locked_test")
        self.assertNotIn("field", json.dumps(event).lower())

    def test_locked_test_requires_and_records_frozen_candidate(self) -> None:
        with self.assertRaises(LockedTestAccessError):
            self.guard.authorize(
                split="locked_test",
                purpose=AccessPurpose.FINAL_EVALUATION,
                run_id="run-002",
            )

        candidate = FrozenCandidate(
            model_sha256="a" * 64,
            config_sha256="b" * 64,
            policy_sha256="c" * 64,
        )
        grant = self.guard.authorize(
            split="locked_test",
            purpose=AccessPurpose.FINAL_EVALUATION,
            run_id="run-003",
            candidate=candidate,
        )

        self.assertTrue(grant.allows("locked_test", AccessPurpose.FINAL_EVALUATION))
        event = self.ledger_events()[-1]
        self.assertEqual(event["decision"], "allowed")
        self.assertEqual(event["model_sha256"], "a" * 64)

    def test_each_training_stage_is_restricted_to_its_split(self) -> None:
        cases = {
            AccessPurpose.FIT: "train",
            AccessPurpose.MODEL_SELECTION: "validation",
            AccessPurpose.CALIBRATION: "calibration",
        }

        for purpose, split in cases.items():
            with self.subTest(purpose=purpose):
                grant = self.guard.authorize(
                    split=split, purpose=purpose, run_id=f"run-{split}"
                )
                self.assertTrue(grant.allows(split, purpose))
                self.assertFalse(grant.allows("locked_test", purpose))


if __name__ == "__main__":
    unittest.main()
