#!/usr/bin/env python3
"""Normalize approved ID datasets and create identity-safe dataset splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tarfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO

SCHEMA_VERSION = "1.0.0"
MIDV_LAIT_ID = "midv-lait-v1-india-nepal-subset"
MIDV_2020_ID = "midv-2020-v1-core"
DEFAULT_SPLIT_RATIOS = {
    "train": 0.60,
    "validation": 0.20,
    "calibration": 0.10,
    "locked_test": 0.10,
}


def allocate_split_counts(
    total: int, ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS
) -> dict[str, int]:
    """Allocate an exact count while representing every split when possible."""
    if total < 0:
        raise ValueError("total cannot be negative")
    if not ratios or any(value < 0 for value in ratios.values()):
        raise ValueError("split ratios must be non-negative")
    ratio_sum = sum(ratios.values())
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")

    names = list(ratios)
    counts = {name: 0 for name in names}
    if total == 0:
        return counts

    if total >= len(names):
        counts = {name: 1 for name in names}
        remaining = total - len(names)
    else:
        remaining = total

    targets = {name: total * ratios[name] for name in names}
    for _ in range(remaining):
        selected = max(
            names, key=lambda name: (targets[name] - counts[name], -names.index(name))
        )
        counts[selected] += 1
    return counts


def assign_identity_splits(
    identities: Iterable[Mapping[str, str]],
    *,
    seed: str,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, str]:
    """Assign whole identities to exact deterministic, stratified splits."""
    by_stratum: dict[str, set[str]] = defaultdict(set)
    for item in identities:
        identity_group = item["identity_group"]
        stratum = item["stratum"]
        by_stratum[stratum].add(identity_group)

    assignments: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        groups = sorted(
            by_stratum[stratum],
            key=lambda group: hashlib.sha256(
                f"{seed}:{stratum}:{group}".encode()
            ).hexdigest(),
        )
        counts = allocate_split_counts(len(groups), ratios)
        offset = 0
        for split in ratios:
            for group in groups[offset : offset + counts[split]]:
                assignments[group] = split
            offset += counts[split]
    return assignments


def validate_no_identity_leakage(records: Iterable[Mapping[str, object]]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        identity_group = str(record["identity_group"])
        split = str(record["split"])
        previous = seen.setdefault(identity_group, split)
        if previous != split:
            detail = (
                f"{identity_group!r} appears in {previous!r} and {split!r}"
            )
            raise ValueError(
                f"identity leakage: {detail}"
            )


def shape_to_polygon(shape: Mapping[str, object]) -> list[list[int | float]]:
    """Convert VIA rectangles or polygons to a common point representation."""
    shape_name = shape.get("name")
    if shape_name == "rect":
        x = shape["x"]
        y = shape["y"]
        width = shape["width"]
        height = shape["height"]
        return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    if shape_name == "polygon":
        x_points = shape["all_points_x"]
        y_points = shape["all_points_y"]
        if not isinstance(x_points, list) or not isinstance(y_points, list):
            raise ValueError("polygon coordinates must be lists")
        if len(x_points) != len(y_points) or len(x_points) < 3:
            raise ValueError("polygon coordinates must contain matching point counts")
        return [[x, y] for x, y in zip(x_points, y_points, strict=True)]
    raise ValueError(f"unsupported VIA shape: {shape_name!r}")


def polygon_out_of_frame(
    polygon: Sequence[Sequence[int | float]], *, width: int, height: int
) -> bool:
    """Report partial/outside annotations without altering source coordinates."""
    for point in polygon:
        if len(point) != 2:
            raise ValueError("polygon points must contain exactly two coordinates")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("polygon coordinates must be finite")
        if x < 0 or y < 0 or x > width or y > height:
            return True
    return False


def approved_dataset_ids(registry_path: Path) -> set[str]:
    """Read the small registry subset needed without adding a YAML dependency."""
    dataset_start = re.compile(r"^\s{2}- id:\s*(\S+)\s*$")
    status_line = re.compile(r"^\s{4}status:\s*(\S+)\s*$")
    approved: set[str] = set()
    current_id: str | None = None
    current_status: str | None = None

    def finish_entry() -> None:
        if current_id is not None and current_status == "approved":
            approved.add(current_id)

    for line in registry_path.read_text(encoding="utf-8").splitlines():
        match = dataset_start.match(line)
        if match:
            finish_entry()
            current_id = match.group(1)
            current_status = None
            continue
        match = status_line.match(line)
        if match and current_id is not None:
            current_status = match.group(1)
    finish_entry()
    return approved


def _jpeg_dimensions(stream: BinaryIO) -> tuple[int, int]:
    if stream.read(2) != b"\xff\xd8":
        raise ValueError("not a JPEG stream")
    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while True:
        byte = stream.read(1)
        if not byte:
            raise ValueError("JPEG has no start-of-frame marker")
        if byte != b"\xff":
            continue
        marker_byte = stream.read(1)
        while marker_byte == b"\xff":
            marker_byte = stream.read(1)
        if not marker_byte:
            raise ValueError("truncated JPEG marker")
        marker = marker_byte[0]
        if marker in {0x01, *range(0xD0, 0xDA)}:
            continue
        length_bytes = stream.read(2)
        if len(length_bytes) != 2:
            raise ValueError("truncated JPEG segment")
        segment_length = int.from_bytes(length_bytes, "big")
        if segment_length < 2:
            raise ValueError("invalid JPEG segment length")
        payload = stream.read(segment_length - 2)
        if len(payload) != segment_length - 2:
            raise ValueError("truncated JPEG segment payload")
        if marker in start_of_frame_markers:
            if len(payload) < 5:
                raise ValueError("truncated JPEG start-of-frame payload")
            return int.from_bytes(payload[3:5], "big"), int.from_bytes(
                payload[1:3], "big"
            )


def image_dimensions(stream: BinaryIO) -> tuple[int, int]:
    signature = stream.read(24)
    stream.seek(0)
    if signature.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(signature) < 24 or signature[12:16] != b"IHDR":
            raise ValueError("invalid PNG header")
        return int.from_bytes(signature[16:20], "big"), int.from_bytes(
            signature[20:24], "big"
        )
    return _jpeg_dimensions(stream)


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def midv_lait_identity_inventory(root: Path) -> list[dict[str, str]]:
    identities: list[dict[str, str]] = []
    for group in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = _midv_lait_metadata(group.name)
        identities.append(
            {
                "identity_group": f"{MIDV_LAIT_ID}:{group.name}",
                "stratum": f"{metadata['document_type']}:{metadata['script']}",
            }
        )
    return identities


def _midv_lait_metadata(group_name: str) -> dict[str, str]:
    aadhaar_match = re.fullmatch(r"ind_aadhaar_([a-z]+)_\d+", group_name)
    if aadhaar_match:
        return {
            "country": "IN",
            "document_type": "IN_AADHAAR",
            "script": aadhaar_match.group(1),
        }
    if re.fullmatch(r"npl_id_\d+", group_name):
        return {"country": "NP", "document_type": "NP_ID_CARD", "script": "mixed"}
    raise ValueError(f"unrecognized MIDV-LAIT document group: {group_name}")


def _canonical_field(source_name: str, document_type: str) -> tuple[str, str]:
    canonical_names = {
        "name": "full_name",
        "name_eng": "full_name",
        "father_name": "parent_name",
        "father_name_eng": "parent_name",
        "birth_date": "date_of_birth",
        "gender": "gender",
        "gender_eng": "gender",
        "number": "id_number",
    }
    canonical_name = canonical_names.get(source_name, source_name)
    if canonical_name == "id_number" and document_type == "IN_AADHAAR":
        pii_type = "IN_AADHAAR_NUMBER"
    elif canonical_name == "id_number" and document_type == "NP_ID_CARD":
        pii_type = "NP_ID_NUMBER"
    else:
        pii_type = {
            "full_name": "PERSON_NAME",
            "parent_name": "RELATED_PERSON_NAME",
            "date_of_birth": "DATE_OF_BIRTH",
            "gender": "GENDER",
        }.get(canonical_name, "IDENTITY_DOCUMENT_FIELD")
    return canonical_name, pii_type


def _normalize_lait_fields(
    annotation: Mapping[str, object], document_type: str
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    fields = annotation.get("fields")
    if not isinstance(fields, list):
        raise ValueError("MIDV-LAIT template annotation must contain a fields list")
    for entry in fields:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(
                "each MIDV-LAIT field entry must contain exactly one field"
            )
        source_name, attributes = next(iter(entry.items()))
        if not isinstance(attributes, dict):
            raise ValueError(f"invalid attributes for field {source_name!r}")
        quad = attributes.get("quad")
        if not isinstance(quad, list) or len(quad) != 4:
            raise ValueError(f"field {source_name!r} must have an xywh quad")
        x, y, width, height = quad
        canonical_name, pii_type = _canonical_field(source_name, document_type)
        normalized.append(
            {
                "source_name": source_name,
                "canonical_name": canonical_name,
                "pii_type": pii_type,
                "polygon": [
                    [x, y],
                    [x + width, y],
                    [x + width, y + height],
                    [x, y + height],
                ],
                "script": attributes.get("script"),
                "value": attributes.get("value"),
            }
        )
    return normalized


def normalize_midv_lait(
    root: Path, split_assignments: Mapping[str, str]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for group in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = _midv_lait_metadata(group.name)
        identity_group = f"{MIDV_LAIT_ID}:{group.name}"
        split = split_assignments[identity_group]
        template_id = f"{identity_group}:template"
        template_path = group / f"{group.name}.png"
        annotation_path = group / f"{group.name}.png.json"
        with template_path.open("rb") as image_stream:
            width, height = image_dimensions(image_stream)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_id": template_id,
                "dataset_id": MIDV_LAIT_ID,
                "identity_group": identity_group,
                "split": split,
                **metadata,
                "modality": "template",
                "image": {
                    "storage": "file",
                    "path": _manifest_path(template_path),
                    "width": width,
                    "height": height,
                    "size_bytes": template_path.stat().st_size,
                },
                "annotations": {
                    "document_polygon": [
                        [0, 0],
                        [width, 0],
                        [width, height],
                        [0, height],
                    ],
                    "document_out_of_frame": False,
                    "fields": _normalize_lait_fields(
                        annotation, metadata["document_type"]
                    ),
                },
                "provenance": {
                    "content_class": "mock_or_modified",
                    "licence": "CC-BY-SA-2.5",
                },
            }
        )

        image_directory = group / f"{group.name}_images"
        groundtruth_directory = group / f"{group.name}_groundtruth"
        for capture_path in sorted(image_directory.glob("*.jpg")):
            groundtruth_path = groundtruth_directory / f"{capture_path.name}.json"
            groundtruth = json.loads(groundtruth_path.read_text(encoding="utf-8"))
            with capture_path.open("rb") as image_stream:
                width, height = image_dimensions(image_stream)
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_id": f"{identity_group}:capture:{capture_path.stem}",
                    "dataset_id": MIDV_LAIT_ID,
                    "identity_group": identity_group,
                    "split": split,
                    **metadata,
                    "modality": "camera_capture",
                    "image": {
                        "storage": "file",
                        "path": _manifest_path(capture_path),
                        "width": width,
                        "height": height,
                        "size_bytes": capture_path.stat().st_size,
                    },
                    "annotations": {
                        "document_polygon": groundtruth["quad"],
                        "document_out_of_frame": polygon_out_of_frame(
                            groundtruth["quad"], width=width, height=height
                        ),
                        "fields": [],
                        "rectification_template_record_id": template_id,
                        "ideal_size": groundtruth["ideal_size"],
                        "document_width_mm": groundtruth["width_mm"],
                    },
                    "provenance": {
                        "content_class": "mock_or_modified",
                        "licence": "CC-BY-SA-2.5",
                    },
                }
            )
    return records


MIDV_2020_DOCUMENTS = {
    "alb_id": ("AL", "AL_ID_CARD"),
    "aze_passport": ("AZ", "AZ_PASSPORT"),
    "esp_id": ("ES", "ES_ID_CARD"),
    "est_id": ("EE", "EE_ID_CARD"),
    "fin_id": ("FI", "FI_ID_CARD"),
    "grc_passport": ("GR", "GR_PASSPORT"),
    "lva_passport": ("LV", "LV_PASSPORT"),
    "rus_internalpassport": ("RU", "RU_INTERNAL_PASSPORT"),
    "srb_passport": ("RS", "RS_PASSPORT"),
    "svk_id": ("SK", "SK_ID_CARD"),
}


def _read_tar_json(archive: tarfile.TarFile, member_name: str) -> dict[str, object]:
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"missing tar member: {member_name}")
    with extracted:
        return json.load(extracted)


def _via_records(annotation: Mapping[str, object]) -> list[dict[str, object]]:
    metadata = annotation.get("_via_img_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("VIA annotation must contain _via_img_metadata")
    records = list(metadata.values())
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("VIA image metadata entries must be objects")
    return sorted(records, key=lambda record: str(record["filename"]))


def _validated_image_filename(value: object) -> str:
    filename = str(value)
    if (
        not filename
        or len(filename) > 255
        or Path(filename).name != filename
        or Path(filename).suffix.lower() not in {".jpg", ".jpeg"}
    ):
        raise ValueError(f"image filename must be a plain JPEG basename: {filename!r}")
    return filename


def midv_2020_identity_inventory(root: Path) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    with tarfile.open(root / "templates.tar", "r") as archive:
        annotation_members = sorted(
            member.name
            for member in archive.getmembers()
            if member.isfile()
            and member.name.startswith("annotations/")
            and member.name.endswith(".json")
        )
        for member_name in annotation_members:
            document_code = Path(member_name).stem
            if document_code not in MIDV_2020_DOCUMENTS:
                raise ValueError(
                    f"unrecognized MIDV-2020 document code: {document_code}"
                )
            annotation = _read_tar_json(archive, member_name)
            for record in _via_records(annotation):
                stem = Path(_validated_image_filename(record["filename"])).stem
                inventory.append(
                    {
                        "identity_group": f"{MIDV_2020_ID}:{document_code}:{stem}",
                        "stratum": MIDV_2020_DOCUMENTS[document_code][1],
                    }
                )
    return inventory


def _canonical_midv_2020_field(source_name: str) -> tuple[str, str]:
    canonical_names = {
        "name": "full_name",
        "surname": "surname",
        "birth_date": "date_of_birth",
        "birth_place": "place_of_birth",
        "issue_date": "issue_date",
        "expiry_date": "expiry_date",
        "gender": "gender",
        "nationality": "nationality",
        "id_number": "id_number",
        "number": "id_number",
        "photo": "face_image",
        "face": "face_image",
        "signature": "signature",
        "authority": "issuing_authority",
    }
    canonical_name = canonical_names.get(source_name, source_name)
    pii_type = {
        "full_name": "PERSON_NAME",
        "surname": "PERSON_NAME",
        "date_of_birth": "DATE_OF_BIRTH",
        "place_of_birth": "PLACE_OF_BIRTH",
        "gender": "GENDER",
        "nationality": "NATIONALITY",
        "id_number": "ID_DOCUMENT_NUMBER",
        "face_image": "FACE_IMAGE",
        "signature": "SIGNATURE",
    }.get(canonical_name, "IDENTITY_DOCUMENT_FIELD")
    return canonical_name, pii_type


def _normalize_via_regions(
    regions: object,
) -> tuple[list[dict[str, object]], list[list[int | float]] | None]:
    if not isinstance(regions, list):
        raise ValueError("VIA regions must be a list")
    fields: list[dict[str, object]] = []
    document_polygon: list[list[int | float]] | None = None
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("VIA region must be an object")
        shape = region.get("shape_attributes")
        attributes = region.get("region_attributes")
        if not isinstance(shape, dict) or not isinstance(attributes, dict):
            raise ValueError("VIA region is missing shape or region attributes")
        source_name = str(attributes.get("field_name", "unknown"))
        polygon = shape_to_polygon(shape)
        if source_name == "doc_quad":
            document_polygon = polygon
            continue
        canonical_name, pii_type = _canonical_midv_2020_field(source_name)
        value = attributes.get("value")
        fields.append(
            {
                "source_name": source_name,
                "canonical_name": canonical_name,
                "pii_type": pii_type,
                "polygon": polygon,
                "script": None,
                "value": value if isinstance(value, (str, int, float)) else None,
            }
        )
    return fields, document_polygon


def normalize_midv_2020(
    root: Path, split_assignments: Mapping[str, str]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    archives = (
        ("template", root / "templates.tar"),
        ("scan_upright", root / "scan_upright.tar"),
        ("scan_rotated", root / "scan_rotated.tar"),
    )
    for modality, archive_path in archives:
        with tarfile.open(archive_path, "r") as archive:
            annotation_members = sorted(
                member.name
                for member in archive.getmembers()
                if member.isfile()
                and member.name.startswith("annotations/")
                and member.name.endswith(".json")
            )
            for annotation_member in annotation_members:
                document_code = Path(annotation_member).stem
                if document_code not in MIDV_2020_DOCUMENTS:
                    raise ValueError(
                        f"unrecognized MIDV-2020 document code: {document_code}"
                    )
                country, document_type = MIDV_2020_DOCUMENTS[document_code]
                annotation = _read_tar_json(archive, annotation_member)
                for via_record in _via_records(annotation):
                    filename = _validated_image_filename(via_record["filename"])
                    identity_group = (
                        f"{MIDV_2020_ID}:{document_code}:{Path(filename).stem}"
                    )
                    image_member_name = f"images/{document_code}/{filename}"
                    image_member = archive.getmember(image_member_name)
                    extracted = archive.extractfile(image_member)
                    if extracted is None:
                        raise FileNotFoundError(
                            f"missing tar member: {image_member_name}"
                        )
                    with extracted:
                        width, height = image_dimensions(extracted)
                    fields, document_polygon = _normalize_via_regions(
                        via_record.get("regions")
                    )
                    if modality == "template":
                        document_polygon = [
                            [0, 0],
                            [width, 0],
                            [width, height],
                            [0, height],
                        ]
                    elif document_polygon is None:
                        raise ValueError(
                            f"{archive_path.name}:{image_member_name} has no doc_quad"
                        )
                    template_id = f"{identity_group}:template"
                    annotations: dict[str, object] = {
                        "document_polygon": document_polygon,
                        "document_out_of_frame": polygon_out_of_frame(
                            document_polygon, width=width, height=height
                        ),
                        "fields": fields,
                    }
                    if modality != "template":
                        annotations["rectification_template_record_id"] = template_id
                    records.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "record_id": f"{identity_group}:{modality}",
                            "dataset_id": MIDV_2020_ID,
                            "identity_group": identity_group,
                            "split": split_assignments[identity_group],
                            "country": country,
                            "document_type": document_type,
                            "script": "latin_or_cyrillic",
                            "modality": modality,
                            "image": {
                                "storage": "tar",
                                "path": _manifest_path(archive_path),
                                "member": image_member_name,
                                "width": width,
                                "height": height,
                                "size_bytes": image_member.size,
                            },
                            "annotations": annotations,
                            "provenance": {
                                "content_class": "mock",
                                "licence": "CC-BY-SA-2.5",
                            },
                        }
                    )
    return records


def validate_record_contract(record: Mapping[str, object]) -> None:
    required = {
        "schema_version",
        "record_id",
        "dataset_id",
        "identity_group",
        "split",
        "country",
        "document_type",
        "script",
        "modality",
        "image",
        "annotations",
        "provenance",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"record is missing required keys: {missing}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {record['schema_version']!r}")
    if record["split"] not in DEFAULT_SPLIT_RATIOS:
        raise ValueError(f"invalid split: {record['split']!r}")
    if not isinstance(record["country"], str) or not re.fullmatch(
        r"[A-Z]{2}", record["country"]
    ):
        raise ValueError(f"invalid ISO alpha-2 country: {record['country']!r}")

    image = record["image"]
    if not isinstance(image, dict):
        raise ValueError("image must be an object")
    image_required = {"storage", "path", "width", "height", "size_bytes"}
    image_missing = sorted(image_required - set(image))
    if image_missing:
        raise ValueError(f"image is missing required keys: {image_missing}")
    if image["storage"] not in {"file", "tar"}:
        raise ValueError(f"invalid image storage: {image['storage']!r}")
    if image["storage"] == "tar" and not image.get("member"):
        raise ValueError("tar image records require a member")
    for key in ("width", "height", "size_bytes"):
        if (
            not isinstance(image[key], int)
            or isinstance(image[key], bool)
            or image[key] <= 0
        ):
            raise ValueError(f"image {key} must be a positive integer")

    annotations = record["annotations"]
    if not isinstance(annotations, dict):
        raise ValueError("annotations must be an object")
    annotations_required = {"document_polygon", "document_out_of_frame", "fields"}
    annotation_missing = sorted(annotations_required - set(annotations))
    if annotation_missing:
        raise ValueError(f"annotations are missing required keys: {annotation_missing}")
    document_polygon = annotations["document_polygon"]
    _validate_polygon(document_polygon, "document_polygon")
    expected_out_of_frame = polygon_out_of_frame(
        document_polygon, width=image["width"], height=image["height"]
    )
    if annotations["document_out_of_frame"] is not expected_out_of_frame:
        record_id = record["record_id"]
        raise ValueError(
            f"document_out_of_frame does not match coordinates for {record_id}"
        )
    fields = annotations["fields"]
    if not isinstance(fields, list):
        raise ValueError("annotations.fields must be a list")
    required_field_keys = {
        "source_name",
        "canonical_name",
        "pii_type",
        "polygon",
        "script",
        "value",
    }
    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            raise ValueError(f"field {index} must be an object")
        field_missing = sorted(required_field_keys - set(field))
        if field_missing:
            raise ValueError(f"field {index} is missing required keys: {field_missing}")
        _validate_polygon(field["polygon"], f"fields[{index}].polygon")


def _validate_polygon(polygon: object, label: str) -> None:
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError(f"{label} must contain at least three points")
    for point in polygon:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{label} points must be two-element arrays")
        for coordinate in point:
            if (
                not isinstance(coordinate, (int, float))
                or isinstance(coordinate, bool)
                or not math.isfinite(coordinate)
            ):
                raise ValueError(f"{label} coordinates must be finite numbers")


def write_processed_dataset(
    records: Sequence[Mapping[str, object]],
    identity_assignments: Mapping[str, str],
    output: Path,
    *,
    seed: str,
) -> dict[str, object]:
    validate_no_identity_leakage(records)
    sorted_records = sorted(records, key=lambda record: str(record["record_id"]))
    seen_record_ids: set[str] = set()
    observed_assignments: dict[str, str] = {}
    for record in sorted_records:
        validate_record_contract(record)
        record_id = str(record["record_id"])
        if record_id in seen_record_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen_record_ids.add(record_id)
        identity_group = str(record["identity_group"])
        split = str(record["split"])
        if identity_assignments.get(identity_group) != split:
            raise ValueError(f"record split disagrees with assignment: {record_id}")
        observed_assignments[identity_group] = split
    if observed_assignments != dict(identity_assignments):
        missing = sorted(set(identity_assignments) - set(observed_assignments))
        extra = sorted(set(observed_assignments) - set(identity_assignments))
        raise ValueError(
            f"identity assignment mismatch; missing={missing}, extra={extra}"
        )

    templates = {
        str(record["record_id"]): str(record["identity_group"])
        for record in sorted_records
        if record["modality"] == "template"
    }
    for record in sorted_records:
        annotations = record["annotations"]
        template_id = annotations.get("rectification_template_record_id")
        if template_id is None:
            continue
        if templates.get(str(template_id)) != record["identity_group"]:
            record_id = record["record_id"]
            raise ValueError(
                f"invalid rectification template link for {record_id}: "
                f"{template_id}"
            )

    output.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []
    records_path = output / "records.jsonl"
    _write_jsonl_atomic(records_path, sorted_records)
    generated_files.append(records_path)
    for split in DEFAULT_SPLIT_RATIOS:
        split_path = output / f"{split}.jsonl"
        _write_jsonl_atomic(
            split_path,
            [record for record in sorted_records if record["split"] == split],
        )
        generated_files.append(split_path)

    assignment_rows = [
        {"identity_group": identity_group, "split": split}
        for identity_group, split in sorted(identity_assignments.items())
    ]
    assignment_path = output / "identity_assignments.jsonl"
    _write_jsonl_atomic(assignment_path, assignment_rows)
    generated_files.append(assignment_path)

    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "split_ratios": DEFAULT_SPLIT_RATIOS,
        "record_count": len(sorted_records),
        "identity_count": len(identity_assignments),
        "records_by_split": _count_by(sorted_records, "split", DEFAULT_SPLIT_RATIOS),
        "identities_by_split": _count_values(
            identity_assignments.values(), DEFAULT_SPLIT_RATIOS
        ),
        "records_by_dataset": _count_by(sorted_records, "dataset_id"),
        "records_by_country": _count_by(sorted_records, "country"),
        "records_by_document_type": _count_by(sorted_records, "document_type"),
        "records_by_modality": _count_by(sorted_records, "modality"),
        "out_of_frame_records": sum(
            bool(record["annotations"].get("document_out_of_frame", False))
            for record in sorted_records
        ),
        "leakage_audit": {
            "cross_split_identities": 0,
            "unit": "identity_group",
            "result": "pass",
        },
        "generated_file_sha256": {
            path.name: _sha256_file(path) for path in generated_files
        },
    }
    _write_json_atomic(output / "build_report.json", report)
    return report


def _count_values(
    values: Iterable[object], required_keys: Iterable[str] = ()
) -> dict[str, int]:
    counts = {key: 0 for key in required_keys}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _count_by(
    records: Iterable[Mapping[str, object]],
    key: str,
    required_keys: Iterable[str] = (),
) -> dict[str, int]:
    return _count_values((record[key] for record in records), required_keys)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
            output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_processed_dataset(
    *,
    registry: Path,
    midv_lait_root: Path,
    midv_2020_root: Path,
    output: Path,
    seed: str,
) -> dict[str, object]:
    approved = approved_dataset_ids(registry)
    required = {MIDV_LAIT_ID, MIDV_2020_ID}
    if not required.issubset(approved):
        missing = sorted(required - approved)
        raise PermissionError(
            f"refusing normalization because datasets are not approved: {missing}"
        )

    inventory = midv_lait_identity_inventory(midv_lait_root)
    inventory.extend(midv_2020_identity_inventory(midv_2020_root))
    assignments = assign_identity_splits(inventory, seed=seed)
    records = normalize_midv_lait(midv_lait_root, assignments)
    records.extend(normalize_midv_2020(midv_2020_root, assignments))
    return write_processed_dataset(records, assignments, output, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize approved MIDV ID datasets into identity-safe JSONL splits."
        )
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("datasets/manifests/registry.yaml")
    )
    parser.add_argument(
        "--midv-lait-root",
        type=Path,
        default=Path("datasets/raw/midv-lait/priority-subset"),
    )
    parser.add_argument(
        "--midv-2020-root", type=Path, default=Path("datasets/raw/midv-2020")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/processed/id-foundation-v1")
    )
    parser.add_argument("--seed", default="pii-id-foundation-v1")
    args = parser.parse_args()

    report = build_processed_dataset(
        registry=args.registry,
        midv_lait_root=args.midv_lait_root,
        midv_2020_root=args.midv_2020_root,
        output=args.output,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
