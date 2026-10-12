from __future__ import annotations

import unittest

from PIL import Image

from pii_id.augmentation.degradations import DegradationAugmenter
from pii_id.models.document import QUALITY_LABELS


class DegradationAugmenterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (100, 60), color=(180, 190, 200))
        self.polygon = [[0, 0], [99, 0], [99, 59], [0, 59]]
        self.augmenter = DegradationAugmenter(seed=17)

    def test_named_degradations_emit_exact_quality_label(self) -> None:
        for name in QUALITY_LABELS:
            with self.subTest(name=name):
                result = self.augmenter.apply_named(
                    self.image, self.polygon, name=name, severity=0.6
                )
                self.assertEqual(result.image.size, self.image.size)
                self.assertEqual(result.labels[QUALITY_LABELS.index(name)], 1.0)
                self.assertEqual(sum(result.labels), 1.0)

    def test_partial_crop_preserves_out_of_frame_coordinates(self) -> None:
        result = self.augmenter.apply_named(
            self.image, self.polygon, name="partial_crop", severity=0.8
        )

        xs = [point[0] for point in result.polygon]
        self.assertLess(min(xs), 0)
        self.assertGreater(max(xs), self.image.width)

    def test_random_augmentation_is_deterministic_per_record_and_epoch(self) -> None:
        first = self.augmenter.apply_random(
            self.image, self.polygon, record_key="record-1", epoch=3
        )
        second = self.augmenter.apply_random(
            self.image, self.polygon, record_key="record-1", epoch=3
        )

        self.assertEqual(first.labels, second.labels)
        self.assertEqual(first.polygon, second.polygon)
        self.assertEqual(first.image.tobytes(), second.image.tobytes())


if __name__ == "__main__":
    unittest.main()
