from __future__ import annotations

import unittest

import torch
from PIL import Image

from pii_id.geometry import canonicalize_quadrilateral, letterbox_document
from pii_id.models.document import (
    QUALITY_LABELS,
    DocumentPerceptionNet,
    document_multitask_loss,
)


class DocumentGeometryTests(unittest.TestCase):
    def test_corner_order_is_canonical_for_rotated_input_order(self) -> None:
        polygon = [[90, 80], [10, 80], [10, 20], [90, 20]]

        ordered = canonicalize_quadrilateral(polygon)

        self.assertEqual(
            ordered,
            [[10.0, 20.0], [90.0, 20.0], [90.0, 80.0], [10.0, 80.0]],
        )

    def test_letterbox_normalizes_long_card_without_clipping_out_of_frame_corner(
        self,
    ) -> None:
        image = Image.new("RGB", (200, 100), color="white")
        polygon = [[-10, 0], [200, 0], [200, 100], [-10, 100]]

        tensor, corners, transform = letterbox_document(image, polygon, size=100)

        self.assertEqual(tuple(tensor.shape), (3, 100, 100))
        self.assertAlmostEqual(transform.scale, 0.5)
        self.assertAlmostEqual(corners[0][0], -0.05)
        self.assertAlmostEqual(corners[0][1], 0.25)
        restored = transform.input_to_source(corners, normalized=True)
        expected_corners = canonicalize_quadrilateral(polygon)
        for actual, expected in zip(restored, expected_corners, strict=True):
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])


class DocumentPerceptionNetTests(unittest.TestCase):
    def test_forward_outputs_all_independent_evidence_heads(self) -> None:
        model = DocumentPerceptionNet(num_document_types=4, pretrained=False)
        batch = torch.rand(2, 3, 96, 96)

        output = model(batch)

        self.assertEqual(tuple(output["presence_logits"].shape), (2,))
        self.assertEqual(tuple(output["corners"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["corner_heatmaps"].shape[:2]), (2, 4))
        self.assertEqual(tuple(output["type_logits"].shape), (2, 4))
        self.assertEqual(
            tuple(output["quality_logits"].shape), (2, len(QUALITY_LABELS))
        )
        self.assertTrue(torch.all(output["corners"] >= -0.5))
        self.assertTrue(torch.all(output["corners"] <= 1.5))

    def test_multitask_loss_is_finite_and_backpropagates(self) -> None:
        model = DocumentPerceptionNet(num_document_types=3, pretrained=False)
        output = model(torch.rand(2, 3, 96, 96))
        targets = {
            "presence": torch.tensor([1.0, 1.0]),
            "corners": torch.tensor(
                [
                    [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]],
                    [[-0.1, 0.1], [1.1, 0.1], [1.1, 0.9], [-0.1, 0.9]],
                ]
            ),
            "document_type": torch.tensor([0, 2]),
            "quality": torch.zeros(2, len(QUALITY_LABELS)),
        }

        losses = document_multitask_loss(output, targets)
        losses["total"].backward()

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(torch.isfinite(losses["corner_heatmap"]))
        self.assertGreater(float(losses["total"].detach()), 0.0)
        has_gradient = any(
            parameter.grad is not None for parameter in model.parameters()
        )
        self.assertTrue(has_gradient)


if __name__ == "__main__":
    unittest.main()
