from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pii_id.data.loader import DatasetLoadError, DatasetManifest


def encoded_image(
    *, image_format: str = "PNG", size: tuple[int, int] = (24, 16)
) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color=(20, 80, 140)).save(output, format=image_format)
    return output.getvalue()


def canonical_record(
    *,
    image_path: str = "images/sample.png",
    storage: str = "file",
    member: str | None = None,
    image_bytes: bytes,
    schema_version: str = "1.0.0",
    dataset_id: str = "approved-set",
) -> dict[str, object]:
    image: dict[str, object] = {
        "storage": storage,
        "path": image_path,
        "width": 24,
        "height": 16,
        "size_bytes": len(image_bytes),
    }
    if member is not None:
        image["member"] = member
    return {
        "schema_version": schema_version,
        "record_id": "record-001",
        "dataset_id": dataset_id,
        "identity_group": "identity-001",
        "split": "train",
        "country": "IN",
        "document_type": "IN_AADHAAR",
        "script": "devanagari",
        "modality": "template",
        "image": image,
        "annotations": {
            "document_polygon": [[0, 0], [23, 0], [23, 15], [0, 15]],
            "document_out_of_frame": False,
            "fields": [
                {
                    "source_name": "number",
                    "canonical_name": "id_number",
                    "pii_type": "IN_AADHAAR_NUMBER",
                    "polygon": [[1, 1], [20, 1], [20, 5], [1, 5]],
                    "script": "latin",
                    "value": "1234 5678 9012",
                }
            ],
        },
        "provenance": {"content_class": "mock", "licence": "test"},
    }


class DatasetManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "datasets" / "manifests").mkdir(parents=True)
        (self.root / "datasets" / "processed").mkdir(parents=True)
        (self.root / "images").mkdir()
        self.registry = self.root / "datasets" / "manifests" / "registry.yaml"
        self.registry.write_text(
            "schema_version: 1\n"
            "datasets:\n"
            "  - id: approved-set\n"
            "    status: approved\n"
            "  - id: quarantined-set\n"
            "    status: quarantined\n",
            encoding="utf-8",
        )
        self.manifest = self.root / "datasets" / "processed" / "train.jsonl"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_manifest(self, record: dict[str, object]) -> None:
        self.manifest.write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def dataset(self, *, max_image_bytes: int = 64 * 1024 * 1024) -> DatasetManifest:
        return DatasetManifest(
            root=self.root,
            manifest_path=self.manifest,
            registry_path=self.registry,
            max_image_bytes=max_image_bytes,
        )

    def test_streams_file_record_and_decodes_declared_image(self) -> None:
        image_bytes = encoded_image()
        (self.root / "images" / "sample.png").write_bytes(image_bytes)
        self.write_manifest(canonical_record(image_bytes=image_bytes))

        records = list(self.dataset().iter_records())
        sample = self.dataset().load_sample(records[0])

        self.assertEqual(len(records), 1)
        self.assertEqual(sample.image.size, (24, 16))
        self.assertEqual(sample.record["document_type"], "IN_AADHAAR")
        self.assertEqual(
            sample.record["annotations"]["fields"][0]["pii_type"],
            "IN_AADHAAR_NUMBER",
        )

    def test_streams_tar_member_without_extracting_archive(self) -> None:
        image_bytes = encoded_image(image_format="JPEG")
        archive_path = self.root / "images" / "samples.tar"
        with tarfile.open(archive_path, "w") as archive:
            member = tarfile.TarInfo("cards/sample.jpg")
            member.size = len(image_bytes)
            archive.addfile(member, io.BytesIO(image_bytes))
        self.write_manifest(
            canonical_record(
                image_path="images/samples.tar",
                storage="tar",
                member="cards/sample.jpg",
                image_bytes=image_bytes,
            )
        )

        record = next(self.dataset().iter_records())
        sample = self.dataset().load_sample(record)

        self.assertEqual(sample.image.size, (24, 16))
        self.assertFalse((self.root / "cards" / "sample.jpg").exists())

    def test_rejects_unknown_schema_and_unapproved_dataset(self) -> None:
        image_bytes = encoded_image()
        (self.root / "images" / "sample.png").write_bytes(image_bytes)
        self.write_manifest(
            canonical_record(image_bytes=image_bytes, schema_version="2.0.0")
        )

        with self.assertRaises(DatasetLoadError) as schema_error:
            next(self.dataset().iter_records())
        self.assertEqual(schema_error.exception.code, "unsupported_schema")

        self.write_manifest(
            canonical_record(image_bytes=image_bytes, dataset_id="quarantined-set")
        )
        with self.assertRaises(DatasetLoadError) as approval_error:
            next(self.dataset().iter_records())
        self.assertEqual(approval_error.exception.code, "dataset_not_approved")

    def test_rejects_file_and_tar_path_traversal(self) -> None:
        image_bytes = encoded_image()
        outside = self.root.parent / "outside.png"
        outside.write_bytes(image_bytes)
        self.addCleanup(outside.unlink, missing_ok=True)
        self.write_manifest(
            canonical_record(image_path="../outside.png", image_bytes=image_bytes)
        )

        record = next(self.dataset().iter_records())
        with self.assertRaises(DatasetLoadError) as path_error:
            self.dataset().load_sample(record)
        self.assertEqual(path_error.exception.code, "unsafe_source_path")

        archive_path = self.root / "images" / "samples.tar"
        with tarfile.open(archive_path, "w") as archive:
            member = tarfile.TarInfo("cards/sample.png")
            member.size = len(image_bytes)
            archive.addfile(member, io.BytesIO(image_bytes))
        self.write_manifest(
            canonical_record(
                image_path="images/samples.tar",
                storage="tar",
                member="../cards/sample.png",
                image_bytes=image_bytes,
            )
        )

        record = next(self.dataset().iter_records())
        with self.assertRaises(DatasetLoadError) as member_error:
            self.dataset().load_sample(record)
        self.assertEqual(member_error.exception.code, "unsafe_archive_member")

    def test_corrupt_and_oversized_images_have_pii_safe_structured_errors(self) -> None:
        corrupt = b"not an image 1234 5678 9012"
        (self.root / "images" / "sample.png").write_bytes(corrupt)
        self.write_manifest(canonical_record(image_bytes=corrupt))
        record = next(self.dataset().iter_records())

        with self.assertRaises(DatasetLoadError) as corrupt_error:
            self.dataset().load_sample(record)

        self.assertEqual(corrupt_error.exception.code, "corrupt_image")
        self.assertNotIn("1234", str(corrupt_error.exception))
        self.assertNotIn("1234", json.dumps(corrupt_error.exception.as_log()))

        valid = encoded_image()
        (self.root / "images" / "sample.png").write_bytes(valid)
        self.write_manifest(canonical_record(image_bytes=valid))
        record = next(self.dataset(max_image_bytes=16).iter_records())
        with self.assertRaises(DatasetLoadError) as size_error:
            self.dataset(max_image_bytes=16).load_sample(record)
        self.assertEqual(size_error.exception.code, "source_too_large")

    def test_audit_checks_all_records_without_retaining_images(self) -> None:
        image_bytes = encoded_image()
        (self.root / "images" / "sample.png").write_bytes(image_bytes)
        record = canonical_record(image_bytes=image_bytes)
        second = canonical_record(image_bytes=image_bytes)
        second["record_id"] = "record-002"
        self.manifest.write_text(
            "\n".join(json.dumps(item) for item in (record, second)) + "\n",
            encoding="utf-8",
        )

        report = self.dataset().audit(decode_images=True)

        self.assertEqual(report.total_records, 2)
        self.assertEqual(report.loaded_images, 2)
        self.assertEqual(report.errors, ())


if __name__ == "__main__":
    unittest.main()
