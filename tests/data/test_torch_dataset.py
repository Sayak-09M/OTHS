from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pii_id.data.loader import DatasetManifest
from pii_id.data.torch_dataset import DocumentLabelSpace, TorchDocumentDataset
from pii_id.evaluation.access import AccessPurpose, SplitAccessGuard


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (40, 20), color="white").save(output, format="PNG")
    return output.getvalue()


class TorchDocumentDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "images").mkdir()
        (self.root / "datasets" / "manifests").mkdir(parents=True)
        (self.root / "datasets" / "processed").mkdir(parents=True)
        data = png_bytes()
        (self.root / "images" / "sample.png").write_bytes(data)
        self.record = {
            "schema_version": "1.0.0",
            "record_id": "record-001",
            "dataset_id": "approved-set",
            "identity_group": "identity-001",
            "split": "train",
            "country": "IN",
            "document_type": "IN_AADHAAR",
            "script": "devanagari",
            "modality": "template",
            "image": {
                "storage": "file",
                "path": "images/sample.png",
                "width": 40,
                "height": 20,
                "size_bytes": len(data),
            },
            "annotations": {
                "document_polygon": [[0, 0], [39, 0], [39, 19], [0, 19]],
                "document_out_of_frame": False,
                "fields": [],
            },
            "provenance": {"content_class": "mock", "licence": "test"},
        }
        self.manifest_path = self.root / "datasets" / "processed" / "train.jsonl"
        self.manifest_path.write_text(json.dumps(self.record) + "\n", encoding="utf-8")
        self.registry_path = self.root / "datasets" / "manifests" / "registry.yaml"
        self.registry_path.write_text(
            "datasets:\n  - id: approved-set\n    status: approved\n",
            encoding="utf-8",
        )
        self.manifest = DatasetManifest(
            root=self.root,
            manifest_path=self.manifest_path,
            registry_path=self.registry_path,
        )
        guard = SplitAccessGuard(
            root=self.root, ledger_path=self.root / "audit" / "access.jsonl"
        )
        self.fit_grant = guard.authorize(
            split="train", purpose=AccessPurpose.FIT, run_id="fit-test"
        )
        self.selection_grant = guard.authorize(
            split="validation",
            purpose=AccessPurpose.MODEL_SELECTION,
            run_id="selection-test",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_label_space_is_deterministic_and_reserves_unknown(self) -> None:
        records = [
            self.record,
            {**self.record, "document_type": "NP_ID_CARD"},
            {**self.record, "document_type": "IN_AADHAAR"},
            {**self.record, "document_type": "NON_ID"},
        ]

        labels = DocumentLabelSpace.from_records(reversed(records))

        self.assertEqual(labels.classes, ("UNKNOWN", "IN_AADHAAR", "NP_ID_CARD"))
        self.assertEqual(labels.index("IN_AADHAAR"), 1)
        self.assertEqual(labels.index("NON_ID"), 0)

    def test_dataset_returns_geometry_and_no_field_values(self) -> None:
        labels = DocumentLabelSpace.from_records([self.record])
        dataset = TorchDocumentDataset(
            manifest=self.manifest,
            grant=self.fit_grant,
            label_space=labels,
            input_size=64,
        )

        sample = dataset[0]

        self.assertEqual(tuple(sample["image"].shape), (3, 64, 64))
        self.assertEqual(tuple(sample["corners"].shape), (4, 2))
        self.assertEqual(sample["document_type"].item(), 1)
        self.assertNotIn("annotations", sample)
        self.assertNotIn("value", json.dumps(sample["metadata"]))

    def test_dataset_refuses_grant_for_another_split(self) -> None:
        labels = DocumentLabelSpace.from_records([self.record])

        with self.assertRaisesRegex(ValueError, "grant"):
            TorchDocumentDataset(
                manifest=self.manifest,
                grant=self.selection_grant,
                label_space=labels,
                input_size=64,
            )

    def test_non_id_record_is_a_negative_presence_example(self) -> None:
        negative = {**self.record, "document_type": "NON_ID"}
        self.manifest_path.write_text(json.dumps(negative) + "\n", encoding="utf-8")
        manifest = DatasetManifest(
            root=self.root,
            manifest_path=self.manifest_path,
            registry_path=self.registry_path,
        )
        labels = DocumentLabelSpace.from_records([negative, self.record])
        dataset = TorchDocumentDataset(
            manifest=manifest,
            grant=self.fit_grant,
            label_space=labels,
            input_size=64,
        )

        sample = dataset[0]

        self.assertEqual(sample["presence"].item(), 0.0)
        self.assertEqual(sample["document_type"].item(), 0)


if __name__ == "__main__":
    unittest.main()
