import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.normalize_and_split import (
    DEFAULT_SPLIT_RATIOS,
    allocate_split_counts,
    approved_dataset_ids,
    assign_identity_splits,
    midv_2020_identity_inventory,
    normalize_midv_2020,
    normalize_midv_lait,
    polygon_out_of_frame,
    shape_to_polygon,
    validate_no_identity_leakage,
    write_processed_dataset,
)


def minimal_png(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )


def minimal_jpeg(width: int, height: int) -> bytes:
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
        + b"\xff\xd9"
    )


def write_midv_2020_tar(path: Path, annotation: dict, image: bytes) -> None:
    with tarfile.open(path, "w") as archive:
        annotation_bytes = json.dumps(annotation).encode("utf-8")
        annotation_member = tarfile.TarInfo("annotations/alb_id.json")
        annotation_member.size = len(annotation_bytes)
        archive.addfile(annotation_member, io.BytesIO(annotation_bytes))

        image_member = tarfile.TarInfo("images/alb_id/00.jpg")
        image_member.size = len(image)
        archive.addfile(image_member, io.BytesIO(image))


class SplitTests(unittest.TestCase):
    def test_exact_counts_keep_all_four_splits_for_small_supported_strata(self) -> None:
        self.assertEqual(
            allocate_split_counts(5, DEFAULT_SPLIT_RATIOS),
            {"train": 2, "validation": 1, "calibration": 1, "locked_test": 1},
        )
        self.assertEqual(
            allocate_split_counts(10, DEFAULT_SPLIT_RATIOS),
            {"train": 6, "validation": 2, "calibration": 1, "locked_test": 1},
        )
        self.assertEqual(
            allocate_split_counts(100, DEFAULT_SPLIT_RATIOS),
            {"train": 60, "validation": 20, "calibration": 10, "locked_test": 10},
        )

    def test_identity_assignment_is_deterministic_and_stratified(self) -> None:
        identities = [
            {"identity_group": f"id-{index:02d}", "stratum": "IN_AADHAAR:devanagari"}
            for index in range(10)
        ]
        first = assign_identity_splits(identities, seed="fixed-seed")
        second = assign_identity_splits(list(reversed(identities)), seed="fixed-seed")

        self.assertEqual(first, second)
        counts = {
            split: list(first.values()).count(split) for split in DEFAULT_SPLIT_RATIOS
        }
        self.assertEqual(
            counts,
            {"train": 6, "validation": 2, "calibration": 1, "locked_test": 1},
        )

    def test_leakage_validator_rejects_one_identity_in_two_splits(self) -> None:
        records = [
            {"identity_group": "same-person", "split": "train"},
            {"identity_group": "same-person", "split": "locked_test"},
        ]

        with self.assertRaisesRegex(ValueError, "identity leakage"):
            validate_no_identity_leakage(records)


class NormalizationTests(unittest.TestCase):
    def test_out_of_frame_annotation_is_flagged_without_clipping_source_truth(
        self,
    ) -> None:
        polygon = [[10, 10], [101, 10], [101, 90], [-2, 90]]

        self.assertTrue(polygon_out_of_frame(polygon, width=100, height=100))
        self.assertEqual(polygon[3], [-2, 90])

    def test_via_rect_and_polygon_shapes_become_four_point_polygons(self) -> None:
        self.assertEqual(
            shape_to_polygon(
                {"name": "rect", "x": 10, "y": 20, "width": 30, "height": 40}
            ),
            [[10, 20], [40, 20], [40, 60], [10, 60]],
        )
        self.assertEqual(
            shape_to_polygon(
                {
                    "name": "polygon",
                    "all_points_x": [1, 5, 5, 1],
                    "all_points_y": [2, 2, 8, 8],
                }
            ),
            [[1, 2], [5, 2], [5, 8], [1, 8]],
        )

    def test_registry_parser_excludes_quarantined_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = Path(temporary_directory) / "registry.yaml"
            registry.write_text(
                "datasets:\n"
                "  - id: safe-one\n"
                "    status: approved\n"
                "  - id: unsafe-one\n"
                "    status: quarantined\n"
                "  - id: safe-two\n"
                "    status: approved\n",
                encoding="utf-8",
            )

            self.assertEqual(approved_dataset_ids(registry), {"safe-one", "safe-two"})

    def test_midv_lait_template_and_capture_share_identity_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            group_name = "ind_aadhaar_devanagari_01"
            group = root / group_name
            images = group / f"{group_name}_images"
            groundtruth = group / f"{group_name}_groundtruth"
            images.mkdir(parents=True)
            groundtruth.mkdir()

            (group / f"{group_name}.png").write_bytes(minimal_png(1011, 646))
            (group / f"{group_name}.png.json").write_text(
                json.dumps(
                    {
                        "fields": [
                            {
                                "number": {
                                    "quad": [100, 200, 300, 40],
                                    "value": "1234 5678 9012",
                                    "script": "latin",
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (images / "000001.jpg").write_bytes(minimal_jpeg(2160, 3840))
            (groundtruth / "000001.jpg.json").write_text(
                json.dumps(
                    {
                        "quad": [[100, 100], [2000, 100], [2000, 3700], [100, 3700]],
                        "ideal_size": [1011, 646],
                        "width_mm": 85.6,
                    }
                ),
                encoding="utf-8",
            )

            identity_group = f"midv-lait-v1-india-nepal-subset:{group_name}"
            records = normalize_midv_lait(root, {identity_group: "locked_test"})

            self.assertEqual(len(records), 2)
            self.assertEqual(
                {record["identity_group"] for record in records}, {identity_group}
            )
            self.assertEqual({record["split"] for record in records}, {"locked_test"})
            template = next(
                record for record in records if record["modality"] == "template"
            )
            capture = next(
                record for record in records if record["modality"] == "camera_capture"
            )
            self.assertEqual(template["image"]["width"], 1011)
            self.assertEqual(capture["image"]["height"], 3840)
            self.assertEqual(
                template["annotations"]["fields"][0]["canonical_name"], "id_number"
            )
            self.assertEqual(
                template["annotations"]["fields"][0]["pii_type"], "IN_AADHAAR_NUMBER"
            )
            self.assertEqual(
                capture["annotations"]["rectification_template_record_id"],
                template["record_id"],
            )

    def test_midv_2020_modalities_share_identity_split_and_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = minimal_jpeg(800, 500)
            template_annotation = {
                "_via_img_metadata": {
                    "00.jpg1": {
                        "filename": "00.jpg",
                        "size": len(image),
                        "regions": [
                            {
                                "shape_attributes": {
                                    "name": "rect",
                                    "x": 20,
                                    "y": 30,
                                    "width": 120,
                                    "height": 25,
                                },
                                "region_attributes": {
                                    "field_name": "number",
                                    "value": "ABC123",
                                },
                            }
                        ],
                    }
                }
            }
            scan_annotation = {
                "_via_img_metadata": {
                    "00.jpg1": {
                        "filename": "00.jpg",
                        "size": len(image),
                        "regions": [
                            {
                                "shape_attributes": {
                                    "name": "polygon",
                                    "all_points_x": [10, 790, 790, 10],
                                    "all_points_y": [10, 10, 490, 490],
                                },
                                "region_attributes": {"field_name": "doc_quad"},
                            }
                        ],
                    }
                }
            }
            write_midv_2020_tar(root / "templates.tar", template_annotation, image)
            write_midv_2020_tar(root / "scan_upright.tar", scan_annotation, image)
            write_midv_2020_tar(root / "scan_rotated.tar", scan_annotation, image)

            inventory = midv_2020_identity_inventory(root)
            self.assertEqual(len(inventory), 1)
            identity_group = inventory[0]["identity_group"]
            records = normalize_midv_2020(root, {identity_group: "validation"})

            self.assertEqual(len(records), 3)
            self.assertEqual(
                {record["identity_group"] for record in records}, {identity_group}
            )
            self.assertEqual({record["split"] for record in records}, {"validation"})
            self.assertEqual(
                {record["modality"] for record in records},
                {"template", "scan_upright", "scan_rotated"},
            )
            template = next(
                record for record in records if record["modality"] == "template"
            )
            scans = [
                record for record in records if record["modality"].startswith("scan_")
            ]
            self.assertEqual(
                template["annotations"]["fields"][0]["canonical_name"], "id_number"
            )
            self.assertTrue(
                all(
                    record["annotations"]["rectification_template_record_id"]
                    == template["record_id"]
                    for record in scans
                )
            )

    def test_midv_2020_rejects_annotation_filenames_with_path_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = minimal_jpeg(20, 10)
            annotation = {
                "_via_img_metadata": {
                    "unsafe": {
                        "filename": "../../00.jpg",
                        "size": len(image),
                        "regions": [],
                    }
                }
            }
            write_midv_2020_tar(root / "templates.tar", annotation, image)

            with self.assertRaisesRegex(ValueError, "plain JPEG basename"):
                midv_2020_identity_inventory(root)

    def test_processed_writer_creates_all_split_manifests_and_audit(self) -> None:
        records = [
            {
                "schema_version": "1.0.0",
                "record_id": "record-1",
                "dataset_id": "dataset-1",
                "identity_group": "identity-1",
                "split": "train",
                "country": "IN",
                "document_type": "IN_AADHAAR",
                "script": "devanagari",
                "modality": "template",
                "image": {
                    "storage": "file",
                    "path": "image.png",
                    "width": 10,
                    "height": 10,
                    "size_bytes": 1,
                },
                "annotations": {
                    "document_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "document_out_of_frame": False,
                    "fields": [],
                },
                "provenance": {"content_class": "mock", "licence": "test"},
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            report = write_processed_dataset(
                records,
                {"identity-1": "train"},
                output,
                seed="fixed-seed",
            )

            self.assertEqual(report["leakage_audit"]["cross_split_identities"], 0)
            self.assertEqual(report["records_by_split"]["train"], 1)
            for split in DEFAULT_SPLIT_RATIOS:
                self.assertTrue((output / f"{split}.jsonl").is_file())
            self.assertTrue((output / "records.jsonl").is_file())
            self.assertTrue((output / "identity_assignments.jsonl").is_file())
            self.assertTrue((output / "build_report.json").is_file())

    def test_processed_writer_rejects_wrong_out_of_frame_flag(self) -> None:
        record = {
            "schema_version": "1.0.0",
            "record_id": "record-1",
            "dataset_id": "dataset-1",
            "identity_group": "identity-1",
            "split": "train",
            "country": "IN",
            "document_type": "IN_AADHAAR",
            "script": "devanagari",
            "modality": "camera_capture",
            "image": {
                "storage": "file",
                "path": "image.png",
                "width": 10,
                "height": 10,
                "size_bytes": 1,
            },
            "annotations": {
                "document_polygon": [[0, 0], [11, 0], [11, 10], [0, 10]],
                "document_out_of_frame": False,
                "fields": [],
            },
            "provenance": {"content_class": "mock", "licence": "test"},
        }
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            self.assertRaisesRegex(ValueError, "document_out_of_frame"),
        ):
            write_processed_dataset(
                [record],
                {"identity-1": "train"},
                Path(temporary_directory),
                seed="fixed-seed",
            )


if __name__ == "__main__":
    unittest.main()
