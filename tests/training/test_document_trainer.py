from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from pii_id.models.document import QUALITY_LABELS, DocumentPerceptionNet
from pii_id.training.document_trainer import (
    evaluate_document_model,
    save_document_checkpoint,
    train_document_epoch,
)


class TinyDocumentDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.full((3, 64, 64), index / 4),
            "presence": torch.tensor(float(index < 3)),
            "corners": torch.tensor([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]),
            "document_type": torch.tensor(index % 2 + 1),
            "quality": torch.zeros(len(QUALITY_LABELS)),
        }


class DocumentTrainerTests(unittest.TestCase):
    def test_training_and_evaluation_run_without_sensitive_metadata(self) -> None:
        torch.manual_seed(7)
        model = DocumentPerceptionNet(num_document_types=3, pretrained=False)
        loader = DataLoader(TinyDocumentDataset(), batch_size=2, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.corner_head.weight.detach().clone()

        training = train_document_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
        )
        evaluation = evaluate_document_model(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.isfinite(torch.tensor(training["loss"])))
        self.assertFalse(torch.equal(before, model.corner_head.weight.detach()))
        self.assertGreaterEqual(evaluation["mean_polygon_iou"], 0.0)
        self.assertLessEqual(evaluation["mean_polygon_iou"], 1.0)
        self.assertEqual(evaluation["positive_examples"], 3.0)
        self.assertEqual(evaluation["negative_examples"], 1.0)
        self.assertGreaterEqual(evaluation["presence_accuracy"], 0.0)
        self.assertLessEqual(evaluation["false_positive_rate"], 1.0)
        self.assertNotIn("metadata", training)

    def test_checkpoint_is_bounded_to_artifact_root_and_hashed(self) -> None:
        model = DocumentPerceptionNet(num_document_types=2, pretrained=False)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = save_document_checkpoint(
                model=model,
                artifact_root=root,
                relative_path=Path("document/model.pt"),
                label_classes=("UNKNOWN", "IN_AADHAAR"),
                config={"input_size": 64},
                metrics={"mean_polygon_iou": 0.5},
            )

            self.assertTrue(result.path.is_file())
            self.assertEqual(len(result.sha256), 64)
            with self.assertRaises(ValueError):
                save_document_checkpoint(
                    model=model,
                    artifact_root=root,
                    relative_path=Path("../escape.pt"),
                    label_classes=("UNKNOWN", "IN_AADHAAR"),
                    config={},
                    metrics={},
                )


if __name__ == "__main__":
    unittest.main()
