from __future__ import annotations

import unittest
from pathlib import Path

from pii_id.cli import _arguments


class CliDefaultsTests(unittest.TestCase):
    def test_default_checkpoint_is_the_trained_foundation_v2_model(self) -> None:
        arguments = _arguments(["input-directory"])

        self.assertEqual(
            arguments.checkpoint, Path("artifacts/document/foundation-v2-best.pt")
        )


if __name__ == "__main__":
    unittest.main()
