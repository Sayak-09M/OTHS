from __future__ import annotations

import io
import json
import math
import re
import tarfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

SCHEMA_VERSION = "1.0.0"
SPLITS = frozenset({"train", "validation", "calibration", "locked_test"})
STORAGE_TYPES = frozenset({"file", "tar"})
IMAGE_FORMATS = frozenset({"BMP", "JPEG", "PNG", "TIFF", "WEBP"})


class DatasetLoadError(Exception):
    """A structured dataset error that never includes annotation values."""

    def __init__(
        self,
        code: str,
        *,
        record_id: str | None = None,
        source: str | None = None,
    ) -> None:
        self.code = code
        self.record_id = record_id
        self.source = source
        super().__init__(code)

    def __str__(self) -> str:
        parts = [self.code]
        if self.record_id:
            parts.append(f"record={self.record_id}")
        if self.source:
            parts.append(f"source={self.source}")
        return " ".join(parts)

    def as_log(self) -> dict[str, str]:
        event = {"error_code": self.code}
        if self.record_id:
            event["record_id"] = self.record_id
        if self.source:
            event["source"] = self.source
        return event


def _safe_record_id(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value[:256]


def _fail(code: str, record: Mapping[str, object], source: str) -> None:
    raise DatasetLoadError(
        code,
        record_id=_safe_record_id(record.get("record_id")),
        source=source,
    )


def _mapping(
    value: object, code: str, record: Mapping[str, object], source: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(code, record, source)
    return value


def _polygon_is_valid(value: object) -> bool:
    if not isinstance(value, list) or len(value) < 3:
        return False
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            return False
        if any(
            not isinstance(coordinate, (int, float))
            or isinstance(coordinate, bool)
            or not math.isfinite(coordinate)
            for coordinate in point
        ):
            return False
    return True


def _polygon_out_of_frame(value: list[list[float]], width: int, height: int) -> bool:
    return any(x < 0 or y < 0 or x > width or y > height for x, y in value)


def validate_record(record: Mapping[str, object], *, source: str) -> None:
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
    if set(record) != required:
        _fail("invalid_record_keys", record, source)
    if record.get("schema_version") != SCHEMA_VERSION:
        _fail("unsupported_schema", record, source)
    if record.get("split") not in SPLITS:
        _fail("invalid_split", record, source)
    if not isinstance(record.get("country"), str) or not re.fullmatch(
        r"[A-Z]{2}", str(record.get("country"))
    ):
        _fail("invalid_country", record, source)
    for key in (
        "record_id",
        "dataset_id",
        "identity_group",
        "document_type",
        "script",
        "modality",
    ):
        if not isinstance(record.get(key), str) or not record[key]:
            _fail(f"invalid_{key}", record, source)

    image = _mapping(record.get("image"), "invalid_image", record, source)
    required_image = {"storage", "path", "width", "height", "size_bytes"}
    allowed_image = required_image | {"member"}
    if not required_image.issubset(image) or not set(image).issubset(allowed_image):
        _fail("invalid_image_keys", record, source)
    if image.get("storage") not in STORAGE_TYPES:
        _fail("invalid_storage", record, source)
    if not isinstance(image.get("path"), str) or not image["path"]:
        _fail("invalid_source_path", record, source)
    if image.get("storage") == "tar" and (
        not isinstance(image.get("member"), str) or not image["member"]
    ):
        _fail("missing_archive_member", record, source)
    for key in ("width", "height", "size_bytes"):
        value = image.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            _fail(f"invalid_image_{key}", record, source)

    annotations = _mapping(
        record.get("annotations"), "invalid_annotations", record, source
    )
    required_annotations = {"document_polygon", "document_out_of_frame", "fields"}
    allowed_annotations = required_annotations | {
        "rectification_template_record_id",
        "ideal_size",
        "document_width_mm",
    }
    if not required_annotations.issubset(annotations) or not set(
        annotations
    ).issubset(allowed_annotations):
        _fail("invalid_annotation_keys", record, source)
    polygon = annotations.get("document_polygon")
    if not _polygon_is_valid(polygon):
        _fail("invalid_document_polygon", record, source)
    out_of_frame = annotations.get("document_out_of_frame")
    if not isinstance(out_of_frame, bool):
        _fail("invalid_out_of_frame_flag", record, source)
    expected = _polygon_out_of_frame(polygon, image["width"], image["height"])
    if out_of_frame is not expected:
        _fail("out_of_frame_mismatch", record, source)
    fields = annotations.get("fields")
    if not isinstance(fields, list):
        _fail("invalid_fields", record, source)
    field_keys = {
        "source_name",
        "canonical_name",
        "pii_type",
        "polygon",
        "script",
        "value",
    }
    for field in fields:
        if not isinstance(field, Mapping) or set(field) != field_keys:
            _fail("invalid_field_keys", record, source)
        if not _polygon_is_valid(field.get("polygon")):
            _fail("invalid_field_polygon", record, source)
        for key in ("source_name", "canonical_name", "pii_type"):
            if not isinstance(field.get(key), str):
                _fail(f"invalid_field_{key}", record, source)
        if field.get("script") is not None and not isinstance(field["script"], str):
            _fail("invalid_field_script", record, source)
        if field.get("value") is not None and not isinstance(
            field["value"], (str, int, float)
        ):
            _fail("invalid_field_value_type", record, source)

    provenance = _mapping(
        record.get("provenance"), "invalid_provenance", record, source
    )
    if set(provenance) != {"content_class", "licence"}:
        _fail("invalid_provenance_keys", record, source)
    if any(not isinstance(provenance.get(key), str) for key in provenance):
        _fail("invalid_provenance", record, source)


def _approved_dataset_ids(registry_path: Path) -> frozenset[str]:
    try:
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DatasetLoadError("invalid_registry", source=registry_path.name) from error
    if not isinstance(registry, Mapping) or not isinstance(
        registry.get("datasets"), list
    ):
        raise DatasetLoadError("invalid_registry", source=registry_path.name)
    approved = {
        entry["id"]
        for entry in registry["datasets"]
        if isinstance(entry, Mapping)
        and entry.get("status") == "approved"
        and isinstance(entry.get("id"), str)
    }
    return frozenset(approved)


@dataclass(frozen=True)
class LoadedSample:
    record: dict[str, Any]
    image: Image.Image


@dataclass(frozen=True)
class AuditReport:
    total_records: int
    loaded_images: int
    errors: tuple[dict[str, str], ...]


class DatasetManifest:
    def __init__(
        self,
        *,
        root: Path,
        manifest_path: Path,
        registry_path: Path,
        max_image_bytes: int = 64 * 1024 * 1024,
        max_image_pixels: int = 100_000_000,
    ) -> None:
        if max_image_bytes <= 0 or max_image_pixels <= 0:
            raise ValueError("image safety limits must be positive")
        self.root = root.resolve(strict=True)
        self.manifest_path = self._resolve_control_path(manifest_path, "manifest")
        self.registry_path = self._resolve_control_path(registry_path, "registry")
        self.max_image_bytes = max_image_bytes
        self.max_image_pixels = max_image_pixels
        self.approved_dataset_ids = _approved_dataset_ids(self.registry_path)

    def _resolve_control_path(self, path: Path, label: str) -> Path:
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise DatasetLoadError(f"{label}_unreadable", source=path.name) from error
        if not resolved.is_relative_to(self.root):
            raise DatasetLoadError(f"unsafe_{label}_path", source=path.name)
        return resolved

    def _source_label(self, record: Mapping[str, object]) -> str:
        image = record.get("image")
        storage = image.get("storage") if isinstance(image, Mapping) else "unknown"
        return f"{storage}:{_safe_record_id(record.get('record_id')) or 'unknown'}"

    def _resolve_image_path(self, record: Mapping[str, object]) -> Path:
        image = record["image"]
        raw_path = Path(image["path"])
        source = self._source_label(record)
        if raw_path.is_absolute():
            _fail("unsafe_source_path", record, source)
        try:
            resolved = (self.root / raw_path).resolve(strict=True)
        except FileNotFoundError as error:
            raise DatasetLoadError(
                "source_missing",
                record_id=_safe_record_id(record.get("record_id")),
                source=source,
            ) from error
        except OSError as error:
            raise DatasetLoadError(
                "source_unreadable",
                record_id=_safe_record_id(record.get("record_id")),
                source=source,
            ) from error
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            _fail("unsafe_source_path", record, source)
        return resolved

    def _validate_declared_size(
        self, record: Mapping[str, object], actual_size: int
    ) -> None:
        source = self._source_label(record)
        declared_size = record["image"]["size_bytes"]
        if declared_size > self.max_image_bytes or actual_size > self.max_image_bytes:
            _fail("source_too_large", record, source)
        if actual_size != declared_size:
            _fail("source_size_mismatch", record, source)

    def _read_file(self, record: Mapping[str, object]) -> bytes:
        path = self._resolve_image_path(record)
        try:
            actual_size = path.stat().st_size
            self._validate_declared_size(record, actual_size)
            return path.read_bytes()
        except DatasetLoadError:
            raise
        except OSError as error:
            raise DatasetLoadError(
                "source_unreadable",
                record_id=_safe_record_id(record.get("record_id")),
                source=self._source_label(record),
            ) from error

    def _validated_member_name(self, record: Mapping[str, object]) -> str:
        member_name = record["image"]["member"]
        member_path = PurePosixPath(member_name)
        if (
            member_path.is_absolute()
            or "\\" in member_name
            or any(part in {"", ".", ".."} for part in member_path.parts)
        ):
            _fail("unsafe_archive_member", record, self._source_label(record))
        return member_name

    def _read_tar(
        self,
        record: Mapping[str, object],
        *,
        archive: tarfile.TarFile | None = None,
    ) -> bytes:
        path = self._resolve_image_path(record)
        member_name = self._validated_member_name(record)
        source = self._source_label(record)
        close_archive = archive is None
        try:
            # The finally block closes archives opened for this one read. A caller-
            # supplied archive remains open so the audit path can reuse it.
            active_archive = archive or tarfile.open(path, mode="r")  # noqa: SIM115
            try:
                member = active_archive.getmember(member_name)
            except KeyError as error:
                raise DatasetLoadError(
                    "archive_member_missing",
                    record_id=_safe_record_id(record.get("record_id")),
                    source=source,
                ) from error
            if not member.isreg():
                _fail("unsafe_archive_member", record, source)
            self._validate_declared_size(record, member.size)
            stream = active_archive.extractfile(member)
            if stream is None:
                _fail("archive_member_unreadable", record, source)
            data = stream.read(self.max_image_bytes + 1)
            if len(data) != member.size:
                _fail("archive_member_truncated", record, source)
            return data
        except DatasetLoadError:
            raise
        except (OSError, tarfile.TarError) as error:
            raise DatasetLoadError(
                "archive_unreadable",
                record_id=_safe_record_id(record.get("record_id")),
                source=source,
            ) from error
        finally:
            if close_archive and "active_archive" in locals():
                active_archive.close()

    def _read_source(
        self,
        record: Mapping[str, object],
        *,
        archive: tarfile.TarFile | None = None,
    ) -> bytes:
        if record["image"]["storage"] == "file":
            return self._read_file(record)
        return self._read_tar(record, archive=archive)

    def _decode_image(
        self, record: Mapping[str, object], image_bytes: bytes
    ) -> Image.Image:
        source = self._source_label(record)
        try:
            with Image.open(io.BytesIO(image_bytes)) as probe:
                if probe.format not in IMAGE_FORMATS:
                    _fail("unsupported_image_format", record, source)
                if probe.size != (
                    record["image"]["width"],
                    record["image"]["height"],
                ):
                    _fail("image_dimension_mismatch", record, source)
                if probe.width * probe.height > self.max_image_pixels:
                    _fail("image_pixel_limit_exceeded", record, source)
                probe.verify()
            with Image.open(io.BytesIO(image_bytes)) as decoded:
                decoded.load()
                return decoded.convert("RGB").copy()
        except DatasetLoadError:
            raise
        except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as error:
            raise DatasetLoadError(
                "corrupt_image",
                record_id=_safe_record_id(record.get("record_id")),
                source=source,
            ) from error

    def iter_records(self) -> Iterator[dict[str, Any]]:
        try:
            manifest = self.manifest_path.open("r", encoding="utf-8")
        except OSError as error:
            raise DatasetLoadError(
                "manifest_unreadable", source=self.manifest_path.name
            ) from error
        with manifest:
            for line_number, line in enumerate(manifest, start=1):
                source = f"{self.manifest_path.name}:{line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetLoadError("invalid_json", source=source) from error
                if not isinstance(record, dict):
                    raise DatasetLoadError("invalid_record", source=source)
                validate_record(record, source=source)
                if record["dataset_id"] not in self.approved_dataset_ids:
                    _fail("dataset_not_approved", record, source)
                yield record

    def load_sample(self, record: Mapping[str, object]) -> LoadedSample:
        source = self._source_label(record)
        validate_record(record, source=source)
        if record["dataset_id"] not in self.approved_dataset_ids:
            _fail("dataset_not_approved", record, source)
        owned_record = dict(record)
        image_bytes = self._read_source(owned_record)
        image = self._decode_image(owned_record, image_bytes)
        return LoadedSample(record=owned_record, image=image)

    def audit(self, *, decode_images: bool = False) -> AuditReport:
        total_records = 0
        loaded_images = 0
        errors: list[dict[str, str]] = []
        archives: dict[Path, tarfile.TarFile] = {}
        try:
            for record in self.iter_records():
                total_records += 1
                try:
                    archive = None
                    if record["image"]["storage"] == "tar":
                        path = self._resolve_image_path(record)
                        archive = archives.get(path)
                        if archive is None:
                            # The outer finally closes every cached archive.
                            archive = tarfile.open(path, mode="r")  # noqa: SIM115
                            archives[path] = archive
                    image_bytes = self._read_source(record, archive=archive)
                    if decode_images:
                        self._decode_image(record, image_bytes)
                    loaded_images += 1
                except DatasetLoadError as error:
                    errors.append(error.as_log())
        finally:
            for archive in archives.values():
                archive.close()
        return AuditReport(
            total_records=total_records,
            loaded_images=loaded_images,
            errors=tuple(errors),
        )
