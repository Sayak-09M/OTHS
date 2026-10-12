from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pii_id.data.loader import DatasetManifest
from pii_id.recognition.identity_profiles import validate_identity_value
from pii_id.recognition.validators import validate_aadhaar
from scripts.build_foundation_v2 import (
    NEGATIVE_DOCUMENT_TYPE,
    POSITIVE_DOCUMENT_TYPES,
    build_foundation_v2,
    generate_identifier,
)


class IdentifierGenerationTests(unittest.TestCase):
    def test_aadhaar_is_format_and_checksum_valid(self) -> None:
        values = {
            generate_identifier("IN_AADHAAR", index, seed="test-seed")
            for index in range(25)
        }

        self.assertEqual(len(values), 25)
        self.assertTrue(all(validate_aadhaar(value) for value in values))

    def test_every_supported_document_type_is_deterministic(self) -> None:
        first = {
            document_type: generate_identifier(document_type, 7, seed="same")
            for document_type in POSITIVE_DOCUMENT_TYPES
        }
        second = {
            document_type: generate_identifier(document_type, 7, seed="same")
            for document_type in reversed(POSITIVE_DOCUMENT_TYPES)
        }

        self.assertEqual(first, second)
        self.assertEqual(len(first["IN_PAN"]), 10)
        self.assertIn(len(first["BD_NID"]), {10, 13, 17})
        self.assertIn(len(first["LK_NIC"]), {10, 12})

    def test_generated_sri_lanka_nics_have_valid_embedded_calendar_days(self) -> None:
        values = {
            generate_identifier("LK_NIC", index, seed="test-seed")
            for index in range(200)
        }

        self.assertEqual(len(values), 200)
        self.assertTrue(
            all(
                validate_identity_value("LK_NIC", value).format_valid
                for value in values
            )
        )


class FoundationBuildTests(unittest.TestCase):
    def test_build_merges_base_records_and_generates_safe_balanced_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "datasets" / "processed" / "id-foundation-v1").mkdir(parents=True)
            (root / "datasets" / "manifests").mkdir(parents=True)
            (root / "images").mkdir()
            base_image = root / "images" / "base.jpg"
            Image.new("RGB", (80, 50), "white").save(base_image, format="JPEG")
            base_record = {
                "schema_version": "1.0.0",
                "record_id": "base-record",
                "dataset_id": "base-approved",
                "identity_group": "base-identity",
                "split": "locked_test",
                "country": "IN",
                "document_type": "IN_AADHAAR",
                "script": "devanagari",
                "modality": "template",
                "image": {
                    "storage": "file",
                    "path": "images/base.jpg",
                    "width": 80,
                    "height": 50,
                    "size_bytes": base_image.stat().st_size,
                },
                "annotations": {
                    "document_polygon": [[0, 0], [79, 0], [79, 49], [0, 49]],
                    "document_out_of_frame": False,
                    "fields": [],
                },
                "provenance": {"content_class": "mock", "licence": "test"},
            }
            base_manifest = (
                root / "datasets" / "processed" / "id-foundation-v1" / "records.jsonl"
            )
            base_manifest.write_text(json.dumps(base_record) + "\n", encoding="utf-8")
            registry = root / "datasets" / "manifests" / "registry.yaml"
            registry.write_text(
                "datasets:\n"
                "  - id: base-approved\n"
                "    status: approved\n"
                "  - id: synthetic-south-asia-id-v2\n"
                "    status: approved\n",
                encoding="utf-8",
            )
            output = root / "datasets" / "processed" / "id-foundation-v2"
            generated = root / "datasets" / "generated" / "synthetic-v2"

            report = build_foundation_v2(
                root=root,
                registry=registry,
                base_manifest=base_manifest,
                output=output,
                generated_root=generated,
                seed="unit-test-v2",
                india_identities_per_type=4,
                regional_identities_per_type=4,
                negative_identities=12,
            )

            records = [
                json.loads(line)
                for line in (output / "records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertIn(base_record, records)
            self.assertEqual(
                next(row for row in records if row["record_id"] == "base-record")[
                    "split"
                ],
                "locked_test",
            )
            self.assertTrue(
                set(POSITIVE_DOCUMENT_TYPES).issubset(
                    {row["document_type"] for row in records}
                )
            )
            self.assertIn(
                NEGATIVE_DOCUMENT_TYPE, {row["document_type"] for row in records}
            )
            self.assertEqual(report["quality_gates"]["identity_leakage"], "pass")
            self.assertEqual(report["quality_gates"]["full_image_decode"], "pass")
            self.assertEqual(report["synthetic"]["real_personal_data"], False)
            self.assertGreater(report["synthetic"]["negative_records"], 0)

            manifest = DatasetManifest(
                root=root,
                manifest_path=output / "records.jsonl",
                registry_path=registry,
            )
            audit = manifest.audit(decode_images=True)
            self.assertEqual(audit.errors, ())
            self.assertEqual(audit.loaded_images, len(records))


if __name__ == "__main__":
    unittest.main()
