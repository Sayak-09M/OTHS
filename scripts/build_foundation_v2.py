#!/usr/bin/env python3
"""Build the privacy-safe, India-priority identity-document foundation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pii_id.data.loader import DatasetManifest  # noqa: E402
from pii_id.recognition.validators import (  # noqa: E402
    aadhaar_check_digit,
    validate_aadhaar,
)
from scripts.normalize_and_split import (  # noqa: E402
    SCHEMA_VERSION,
    approved_dataset_ids,
    assign_identity_splits,
    validate_no_identity_leakage,
    write_processed_dataset,
)

SYNTHETIC_DATASET_ID = "synthetic-south-asia-id-v2"
NEGATIVE_DOCUMENT_TYPE = "NON_ID"
INDIA_DOCUMENT_TYPES = (
    "IN_AADHAAR",
    "IN_PAN",
    "IN_EPIC",
    "IN_ABHA",
    "IN_PASSPORT",
    "IN_DRIVING_LICENCE",
)
REGIONAL_DOCUMENT_TYPES = ("NP_ID_CARD", "BD_NID", "LK_NIC")
POSITIVE_DOCUMENT_TYPES = INDIA_DOCUMENT_TYPES + REGIONAL_DOCUMENT_TYPES
NEGATIVE_KINDS = (
    "hospital_admission_form",
    "hospital_bill",
    "lab_report",
    "hotel_registration_form",
    "hotel_invoice",
    "spreadsheet_export",
)
VARIANTS = ("template", "flat_scan", "camera_rotated", "camera_degraded")

_CARD_WIDTH = 1000
_CARD_HEIGHT = 630
_LICENCE = "Project-generated fictional synthetic data; internal use"


@dataclass(frozen=True)
class DocumentSpec:
    country: str
    title: str
    subtitle: str
    accent: tuple[int, int, int]
    assurance: str


DOCUMENT_SPECS = {
    "IN_AADHAAR": DocumentSpec(
        "IN",
        "INDIA ID RESEARCH CARD",
        "AADHAAR-LIKE SYNTHETIC SAMPLE",
        (230, 120, 55),
        "official_length_and_verhoeff_checksum",
    ),
    "IN_PAN": DocumentSpec(
        "IN",
        "INDIA TAX ID RESEARCH CARD",
        "PAN-LIKE SYNTHETIC SAMPLE",
        (38, 92, 128),
        "official_character_structure_no_public_checksum_validation",
    ),
    "IN_EPIC": DocumentSpec(
        "IN",
        "INDIA ELECTOR ID RESEARCH CARD",
        "EPIC-LIKE SYNTHETIC SAMPLE",
        (89, 94, 154),
        "official_ten_character_alphanumeric_structure",
    ),
    "IN_ABHA": DocumentSpec(
        "IN",
        "INDIA HEALTH ID RESEARCH CARD",
        "ABHA-LIKE SYNTHETIC SAMPLE",
        (31, 139, 121),
        "official_fourteen_digit_length",
    ),
    "IN_PASSPORT": DocumentSpec(
        "IN",
        "INDIA TRAVEL ID RESEARCH PAGE",
        "PASSPORT-LIKE SYNTHETIC SAMPLE",
        (74, 83, 54),
        "candidate_detector_pattern_not_officially_checksum_verified",
    ),
    "IN_DRIVING_LICENCE": DocumentSpec(
        "IN",
        "INDIA LICENCE RESEARCH CARD",
        "DL-LIKE SYNTHETIC SAMPLE",
        (33, 112, 151),
        "candidate_printed_pattern_with_official_smart_card_field_capacity",
    ),
    "NP_ID_CARD": DocumentSpec(
        "NP",
        "NEPAL ID RESEARCH CARD",
        "NATIONAL-ID-LIKE SYNTHETIC SAMPLE",
        (176, 40, 50),
        "official_ten_digit_length",
    ),
    "BD_NID": DocumentSpec(
        "BD",
        "BANGLADESH ID RESEARCH CARD",
        "NID-LIKE SYNTHETIC SAMPLE",
        (20, 128, 80),
        "government_supported_legacy_and_current_lengths",
    ),
    "LK_NIC": DocumentSpec(
        "LK",
        "SRI LANKA ID RESEARCH CARD",
        "NIC-LIKE SYNTHETIC SAMPLE",
        (125, 63, 152),
        "official_legacy_and_current_lengths",
    ),
}


def _digest(seed: str, document_type: str, index: int) -> bytes:
    return hashlib.sha256(f"{seed}:{document_type}:{index}".encode()).digest()


def _digits(digest: bytes, length: int, *, offset: int = 0) -> str:
    stream = "".join(f"{byte:03d}" for byte in digest)
    while len(stream) < offset + length:
        stream += hashlib.sha256(stream.encode()).hexdigest()
        stream = "".join(str(int(character, 16) % 10) for character in stream)
    return stream[offset : offset + length]


def _letters(digest: bytes, length: int, *, offset: int = 0) -> str:
    return "".join(
        chr(ord("A") + digest[(offset + index) % len(digest)] % 26)
        for index in range(length)
    )


def generate_identifier(document_type: str, index: int, *, seed: str) -> str:
    """Return a deterministic fictional identifier matching the declared format."""
    if document_type not in POSITIVE_DOCUMENT_TYPES:
        raise ValueError(f"unsupported document type: {document_type}")
    if index < 0:
        raise ValueError("index must be non-negative")
    digest = _digest(seed, document_type, index)
    if document_type == "IN_AADHAAR":
        base = str(2 + digest[0] % 8) + _digits(digest, 10, offset=1)
        return base + aadhaar_check_digit(base)
    if document_type == "IN_PAN":
        prefix = f"{_letters(digest, 3)}PS"
        return f"{prefix}{_digits(digest, 4, offset=4)}{_letters(digest, 1, offset=8)}"
    if document_type == "IN_EPIC":
        return f"{_letters(digest, 3)}{_digits(digest, 7, offset=3)}"
    if document_type == "IN_ABHA":
        return _digits(digest, 14)
    if document_type == "IN_PASSPORT":
        return f"{_letters(digest, 1)}{_digits(digest, 7, offset=1)}"
    if document_type == "IN_DRIVING_LICENCE":
        prefix = f"{_letters(digest, 2)}{_digits(digest, 2, offset=2)}2026"
        return f"{prefix}{_digits(digest, 7, offset=4)}"
    if document_type == "NP_ID_CARD":
        return _digits(digest, 10)
    if document_type == "BD_NID":
        length = (10, 13, 17)[index % 3]
        return _digits(digest, length)
    calendar_day = 1 + int.from_bytes(digest[:2], "big") % 366
    encoded_day = calendar_day + (500 if digest[2] % 2 else 0)
    if index % 2:
        birth_year = 40 + digest[3] % 60
        serial = _digits(digest, 4, offset=4)
        suffix = "V" if digest[0] % 2 else "X"
        return f"{birth_year:02d}{encoded_day:03d}{serial}{suffix}"
    birth_year = 1940 + digest[3] % 67
    serial = _digits(digest, 5, offset=4)
    return f"{birth_year:04d}{encoded_day:03d}{serial}"


def _font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _box(x: int, y: int, width: int, height: int) -> list[list[float]]:
    return [
        [float(x), float(y)],
        [float(x + width), float(y)],
        [float(x + width), float(y + height)],
        [float(x), float(y + height)],
    ]


def _field(
    source_name: str,
    canonical_name: str,
    pii_type: str,
    polygon: list[list[float]],
    value: str,
    *,
    script: str | None = "latin",
) -> dict[str, object]:
    return {
        "source_name": source_name,
        "canonical_name": canonical_name,
        "pii_type": pii_type,
        "polygon": polygon,
        "script": script,
        "value": value,
    }


def _draw_avatar(draw: ImageDraw.ImageDraw, *, index: int) -> None:
    background = (225 - index % 30, 232, 237)
    draw.rounded_rectangle(
        (58, 168, 322, 500), radius=18, fill=background, outline=(70, 78, 86), width=3
    )
    skin = (176 + index % 45, 126 + index % 35, 94 + index % 30)
    draw.ellipse((128, 205, 252, 329), fill=skin, outline=(70, 60, 55), width=3)
    draw.pieslice((91, 315, 289, 520), 180, 360, fill=(60 + index % 80, 91, 143))
    draw.arc((143, 250, 237, 309), 10, 170, fill=(75, 55, 45), width=3)
    draw.ellipse((158, 259, 166, 267), fill=(35, 35, 35))
    draw.ellipse((214, 259, 222, 267), fill=(35, 35, 35))


def _render_identity_card(
    document_type: str, index: int, *, seed: str
) -> tuple[Image.Image, list[dict[str, object]], str]:
    spec = DOCUMENT_SPECS[document_type]
    identifier = generate_identifier(document_type, index, seed=seed)
    image = Image.new("RGB", (_CARD_WIDTH, _CARD_HEIGHT), (247, 247, 242))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, 997, 627), radius=28, outline=(45, 48, 52), width=5)
    draw.rounded_rectangle((3, 3, 996, 128), radius=26, fill=spec.accent)
    draw.rectangle((3, 95, 996, 128), fill=spec.accent)
    draw.text((42, 22), spec.title, font=_font(34, bold=True), fill="white")
    draw.text((43, 72), spec.subtitle, font=_font(20), fill=(250, 250, 250))
    draw.text((846, 37), spec.country, font=_font(46, bold=True), fill="white")
    _draw_avatar(draw, index=index)
    name = f"SYNTHETIC PERSON {index:05d}"
    date_of_birth = f"{1 + index % 27:02d}/{1 + index % 12:02d}/{1970 + index % 35}"
    address = f"TEST WARD {1 + index % 90}, DEMO DISTRICT"
    rows = (
        ("NAME", name, 370, 166, 30),
        ("IDENTIFIER", identifier, 370, 278, 44),
        ("DATE OF BIRTH", date_of_birth, 370, 394, 29),
        ("ADDRESS", address, 370, 488, 25),
    )
    for label, value, x, y, size in rows:
        draw.text((x, y), label, font=_font(16, bold=True), fill=(82, 85, 88))
        draw.text((x, y + 26), value, font=_font(size, bold=True), fill=(30, 32, 34))
    for x in range(-120, 1080, 255):
        draw.text(
            (x, 568),
            "SYNTHETIC / NOT VALID",
            font=_font(18, bold=True),
            fill=(163, 50, 50),
        )
    draw.text(
        (59, 520), "GEOMETRIC AVATAR", font=_font(14, bold=True), fill=(72, 75, 78)
    )
    fields = [
        _field(
            "portrait",
            "face_image",
            "FACE_IMAGE",
            _box(58, 168, 264, 332),
            "",
            script=None,
        ),
        _field("name", "full_name", "PERSON_NAME", _box(370, 192, 555, 46), name),
        _field(
            "identifier",
            "id_number",
            f"{document_type}_NUMBER",
            _box(370, 304, 570, 58),
            identifier,
        ),
        _field(
            "date_of_birth",
            "date_of_birth",
            "DATE_OF_BIRTH",
            _box(370, 420, 320, 42),
            date_of_birth,
        ),
        _field(
            "address", "address", "POSTAL_ADDRESS", _box(370, 514, 570, 36), address
        ),
    ]
    return image, fields, identifier


def _invalid_aadhaar_like(seed: str, index: int) -> str:
    valid = generate_identifier("IN_AADHAAR", index, seed=f"negative:{seed}")
    replacement = str((int(valid[-1]) + 1) % 10)
    candidate = valid[:-1] + replacement
    if validate_aadhaar(candidate):
        raise AssertionError("hard-negative number unexpectedly passed checksum")
    return candidate


def _render_negative_document(
    kind: str, index: int, *, seed: str
) -> tuple[Image.Image, list[dict[str, object]], None]:
    if kind not in NEGATIVE_KINDS:
        raise ValueError(f"unsupported negative kind: {kind}")
    image = Image.new("RGB", (_CARD_WIDTH, _CARD_HEIGHT), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    palette = (
        (42, 112, 124)
        if kind.startswith("hospital") or kind == "lab_report"
        else (119, 83, 48)
    )
    draw.rectangle((0, 0, _CARD_WIDTH, 94), fill=palette)
    title = kind.replace("_", " ").upper()
    draw.text((38, 22), f"SYNTHETIC {title}", font=_font(34, bold=True), fill="white")
    draw.text(
        (42, 112),
        "TRAINING FORM — NOT AN IDENTITY DOCUMENT",
        font=_font(22, bold=True),
        fill=(170, 45, 45),
    )
    digest = _digest(seed, kind, index)
    lines = (
        ("Name", f"SYNTHETIC GUEST {index:05d}"),
        ("Record / booking number", _digits(digest, 14)),
        (
            "Invoice reference",
            f"{_letters(digest, 5)}{_digits(digest, 4)}{_letters(digest, 1, offset=8)}",
        ),
        ("Long numeric reference", _invalid_aadhaar_like(seed, index)),
        ("Email", f"fictional-{index:05d}@example.invalid"),
        ("Phone", f"+91 90000 {_digits(digest, 5, offset=5)}"),
        ("Notes", "DEMO DATA ONLY — NO GOVERNMENT ID ATTACHED"),
    )
    y = 172
    for label, value in lines:
        draw.text((55, y), label.upper(), font=_font(15, bold=True), fill=(87, 89, 91))
        draw.text((330, y - 3), value, font=_font(24, bold=True), fill=(28, 31, 33))
        draw.line((50, y + 37, 950, y + 37), fill=(195, 198, 200), width=2)
        y += 58
    draw.rectangle((54, 574, 946, 612), outline=palette, width=3)
    draw.text(
        (175, 580),
        "SYNTHETIC HARD NEGATIVE / NOT VALID",
        font=_font(22, bold=True),
        fill=(170, 45, 45),
    )
    return image, [], None


def _scene_background(width: int, height: int, generator: random.Random) -> Image.Image:
    base = 178 + generator.randrange(35)
    image = Image.new("RGB", (width, height), (base, base + 2, base + 4))
    draw = ImageDraw.Draw(image)
    for y in range(0, height, 48):
        shade = base - 5 + (y // 48) % 10
        draw.line((0, y, width, y), fill=(shade, shade + 2, shade + 4), width=2)
    return image


def _map_polygon(
    polygon: Sequence[Sequence[float]],
    *,
    source_size: tuple[int, int],
    resized_size: tuple[int, int],
    angle: float,
    rotated_size: tuple[int, int],
    offset: tuple[int, int],
) -> list[list[float]]:
    source_width, source_height = source_size
    resized_width, resized_height = resized_size
    rotated_width, rotated_height = rotated_size
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    result: list[list[float]] = []
    for point in polygon:
        x = float(point[0]) * resized_width / source_width
        y = float(point[1]) * resized_height / source_height
        centered_x = x - resized_width / 2
        centered_y = y - resized_height / 2
        rotated_x = cosine * centered_x + sine * centered_y
        rotated_y = -sine * centered_x + cosine * centered_y
        result.append(
            [
                rotated_x + rotated_width / 2 + offset[0],
                rotated_y + rotated_height / 2 + offset[1],
            ]
        )
    return result


def _transform_fields(
    fields: Sequence[Mapping[str, object]],
    **transform: object,
) -> list[dict[str, object]]:
    transformed: list[dict[str, object]] = []
    for field in fields:
        transformed.append(
            {
                **field,
                "polygon": _map_polygon(field["polygon"], **transform),
            }
        )
    return transformed


def _render_variant(
    card: Image.Image,
    fields: Sequence[Mapping[str, object]],
    *,
    modality: str,
    generator: random.Random,
) -> tuple[Image.Image, list[list[float]], list[dict[str, object]]]:
    if modality == "template":
        polygon = _box(0, 0, card.width - 1, card.height - 1)
        return card.copy(), polygon, [dict(field) for field in fields]
    if modality == "flat_scan":
        canvas = _scene_background(1280, 860, generator)
        resized_size = (1100, 693)
        resized = card.resize(resized_size, Image.Resampling.LANCZOS)
        offset = (90, 83)
        canvas.paste(resized, offset)
        transform = {
            "source_size": card.size,
            "resized_size": resized_size,
            "angle": 0.0,
            "rotated_size": resized_size,
            "offset": offset,
        }
        polygon = _map_polygon(_box(0, 0, card.width - 1, card.height - 1), **transform)
        return canvas, polygon, _transform_fields(fields, **transform)
    if modality == "camera_rotated":
        canvas = _scene_background(1400, 1000, generator)
        resized_size = (1000, 630)
        resized = card.resize(resized_size, Image.Resampling.LANCZOS).convert("RGBA")
        angle = generator.choice((-10.0, -7.0, 6.0, 9.0))
        rotated = resized.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        offset = (
            (canvas.width - rotated.width) // 2,
            (canvas.height - rotated.height) // 2,
        )
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow.paste(
            (0, 0, 0, 120), (offset[0] + 16, offset[1] + 20), rotated.getchannel("A")
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(14))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow)
        canvas.alpha_composite(rotated, dest=offset)
        transform = {
            "source_size": card.size,
            "resized_size": resized_size,
            "angle": angle,
            "rotated_size": rotated.size,
            "offset": offset,
        }
        polygon = _map_polygon(_box(0, 0, card.width - 1, card.height - 1), **transform)
        return canvas.convert("RGB"), polygon, _transform_fields(fields, **transform)
    if modality != "camera_degraded":
        raise ValueError(f"unsupported modality: {modality}")
    canvas = _scene_background(1280, 860, generator)
    resized_size = (1160, 731)
    resized = card.resize(resized_size, Image.Resampling.LANCZOS)
    offset = (-45, 78) if generator.random() < 0.5 else (165, 76)
    canvas.paste(resized, offset)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.polygon(((0, 0), (520, 0), (230, 860), (0, 860)), fill=(0, 0, 0, 62))
    draw.ellipse((720, 80, 1260, 520), fill=(255, 255, 255, 92))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    canvas = ImageEnhance.Contrast(canvas).enhance(0.88)
    canvas = canvas.resize((640, 430), Image.Resampling.BILINEAR).resize(
        (1280, 860), Image.Resampling.BILINEAR
    )
    canvas = canvas.filter(ImageFilter.GaussianBlur(0.65))
    transform = {
        "source_size": card.size,
        "resized_size": resized_size,
        "angle": 0.0,
        "rotated_size": resized_size,
        "offset": offset,
    }
    polygon = _map_polygon(_box(0, 0, card.width - 1, card.height - 1), **transform)
    return canvas, polygon, _transform_fields(fields, **transform)


def _save_jpeg_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        image.convert("RGB").save(temporary, format="JPEG", quality=88, subsampling=0)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_out_of_frame(
    polygon: Sequence[Sequence[float]], *, width: int, height: int
) -> bool:
    return any(x < 0 or y < 0 or x > width or y > height for x, y in polygon)


def _record(
    *,
    root: Path,
    image_path: Path,
    identity_group: str,
    split: str,
    country: str,
    document_type: str,
    modality: str,
    polygon: list[list[float]],
    fields: list[dict[str, object]],
    template_record_id: str | None,
) -> dict[str, object]:
    relative_path = image_path.resolve(strict=True).relative_to(root).as_posix()
    with Image.open(image_path) as probe:
        width, height = probe.size
        probe.verify()
    annotations: dict[str, object] = {
        "document_polygon": polygon,
        "document_out_of_frame": _is_out_of_frame(polygon, width=width, height=height),
        "fields": fields,
    }
    if template_record_id is not None:
        annotations["rectification_template_record_id"] = template_record_id
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"{identity_group}:{modality}",
        "dataset_id": SYNTHETIC_DATASET_ID,
        "identity_group": identity_group,
        "split": split,
        "country": country,
        "document_type": document_type,
        "script": "latin",
        "modality": modality,
        "image": {
            "storage": "file",
            "path": relative_path,
            "width": width,
            "height": height,
            "size_bytes": image_path.stat().st_size,
        },
        "annotations": annotations,
        "provenance": {
            "content_class": "project_generated_fictional_synthetic",
            "licence": _LICENCE,
        },
    }


def _synthetic_inventory(
    *,
    india_identities_per_type: int,
    regional_identities_per_type: int,
    negative_identities: int,
) -> list[dict[str, str]]:
    values = (
        india_identities_per_type,
        regional_identities_per_type,
        negative_identities,
    )
    if any(value <= 0 for value in values):
        raise ValueError("all identity counts must be positive")
    inventory: list[dict[str, str]] = []
    for document_type in POSITIVE_DOCUMENT_TYPES:
        count = (
            india_identities_per_type
            if document_type in INDIA_DOCUMENT_TYPES
            else regional_identities_per_type
        )
        for index in range(count):
            identity_group = f"{SYNTHETIC_DATASET_ID}:{document_type}:{index:05d}"
            inventory.append(
                {
                    "identity_group": identity_group,
                    "stratum": f"{document_type}:latin",
                }
            )
    for index in range(negative_identities):
        kind = NEGATIVE_KINDS[index % len(NEGATIVE_KINDS)]
        identity_group = (
            f"{SYNTHETIC_DATASET_ID}:{NEGATIVE_DOCUMENT_TYPE}:{kind}:{index:05d}"
        )
        inventory.append(
            {
                "identity_group": identity_group,
                "stratum": f"{NEGATIVE_DOCUMENT_TYPE}:{kind}",
            }
        )
    return inventory


def _generate_synthetic_records(
    *,
    root: Path,
    generated_root: Path,
    assignments: Mapping[str, str],
    seed: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for identity_group in sorted(assignments):
        parts = identity_group.split(":")
        if parts[1] == NEGATIVE_DOCUMENT_TYPE:
            document_type = NEGATIVE_DOCUMENT_TYPE
            kind = parts[2]
            index = int(parts[3])
            country = "IN"
            card, fields, _ = _render_negative_document(kind, index, seed=seed)
            folder = generated_root / document_type / kind / f"{index:05d}"
        else:
            document_type = parts[1]
            index = int(parts[2])
            country = DOCUMENT_SPECS[document_type].country
            card, fields, _ = _render_identity_card(document_type, index, seed=seed)
            folder = generated_root / document_type / f"{index:05d}"
        template_record_id = f"{identity_group}:template"
        identity_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{identity_group}".encode()).digest()[:8], "big"
        )
        for variant_index, modality in enumerate(VARIANTS):
            generator = random.Random(identity_seed + variant_index)
            image, polygon, transformed_fields = _render_variant(
                card, fields, modality=modality, generator=generator
            )
            image_path = folder / f"{modality}.jpg"
            _save_jpeg_atomic(image, image_path)
            records.append(
                _record(
                    root=root,
                    image_path=image_path,
                    identity_group=identity_group,
                    split=assignments[identity_group],
                    country=country,
                    document_type=document_type,
                    modality=modality,
                    polygon=polygon,
                    fields=transformed_fields,
                    template_record_id=(
                        None if modality == "template" else template_record_id
                    ),
                )
            )
    return records


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid base manifest JSON at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"invalid base manifest record at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("base manifest must not be empty")
    return rows


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _inside_root(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} must stay inside the project root")
    return resolved


def build_foundation_v2(
    *,
    root: Path,
    registry: Path,
    base_manifest: Path,
    output: Path,
    generated_root: Path,
    seed: str,
    india_identities_per_type: int,
    regional_identities_per_type: int,
    negative_identities: int,
) -> dict[str, object]:
    """Merge v1 with generated examples and enforce all v2 quality gates."""
    project_root = root.resolve(strict=True)
    registry = _inside_root(registry, project_root, "registry").resolve(strict=True)
    base_manifest = _inside_root(base_manifest, project_root, "base manifest").resolve(
        strict=True
    )
    output = _inside_root(output, project_root, "output")
    generated_root = _inside_root(generated_root, project_root, "generated root")
    approved = approved_dataset_ids(registry)
    if SYNTHETIC_DATASET_ID not in approved:
        raise PermissionError(f"{SYNTHETIC_DATASET_ID} is not approved in the registry")

    base_records = _read_jsonl(base_manifest)
    base_dataset_ids = {str(record["dataset_id"]) for record in base_records}
    if not base_dataset_ids.issubset(approved):
        raise PermissionError(
            f"base datasets are not approved: {sorted(base_dataset_ids - approved)}"
        )
    base_assignments: dict[str, str] = {}
    for record in base_records:
        identity_group = str(record["identity_group"])
        split = str(record["split"])
        previous = base_assignments.setdefault(identity_group, split)
        if previous != split:
            raise ValueError(f"base identity leakage: {identity_group}")

    inventory = _synthetic_inventory(
        india_identities_per_type=india_identities_per_type,
        regional_identities_per_type=regional_identities_per_type,
        negative_identities=negative_identities,
    )
    synthetic_assignments = assign_identity_splits(inventory, seed=seed)
    synthetic_records = _generate_synthetic_records(
        root=project_root,
        generated_root=generated_root,
        assignments=synthetic_assignments,
        seed=seed,
    )
    records = [*base_records, *synthetic_records]
    assignments = {**base_assignments, **synthetic_assignments}
    validate_no_identity_leakage(records)

    values_by_type: dict[str, set[str]] = {
        document_type: set() for document_type in POSITIVE_DOCUMENT_TYPES
    }
    for record in synthetic_records:
        if (
            record["modality"] != "template"
            or record["document_type"] == NEGATIVE_DOCUMENT_TYPE
        ):
            continue
        id_fields = [
            field
            for field in record["annotations"]["fields"]
            if field["canonical_name"] == "id_number"
        ]
        if len(id_fields) != 1:
            raise ValueError(f"record has no unique ID field: {record['record_id']}")
        values_by_type[str(record["document_type"])].add(str(id_fields[0]["value"]))
    expected_identity_counts = {
        document_type: (
            india_identities_per_type
            if document_type in INDIA_DOCUMENT_TYPES
            else regional_identities_per_type
        )
        for document_type in POSITIVE_DOCUMENT_TYPES
    }
    for document_type, expected in expected_identity_counts.items():
        if len(values_by_type[document_type]) != expected:
            raise ValueError(f"non-unique synthetic values for {document_type}")
    if not all(validate_aadhaar(value) for value in values_by_type["IN_AADHAAR"]):
        raise ValueError("generated Aadhaar values failed Verhoeff validation")

    report = write_processed_dataset(records, assignments, output, seed=seed)
    manifest = DatasetManifest(
        root=project_root,
        manifest_path=output / "records.jsonl",
        registry_path=registry,
    )
    audit = manifest.audit(decode_images=True)
    if audit.errors or audit.loaded_images != len(records):
        error_count = len(audit.errors)
        raise RuntimeError(
            f"full decode failed: loaded={audit.loaded_images} errors={error_count}"
        )
    synthetic_counts = Counter(
        str(record["document_type"]) for record in synthetic_records
    )
    report.update(
        {
            "foundation_version": "2.0.0",
            "base_manifest": base_manifest.relative_to(project_root).as_posix(),
            "base_split_policy": "preserved_without_reassignment",
            "synthetic": {
                "dataset_id": SYNTHETIC_DATASET_ID,
                "records": len(synthetic_records),
                "identities": len(synthetic_assignments),
                "records_by_document_type": dict(sorted(synthetic_counts.items())),
                "negative_records": synthetic_counts[NEGATIVE_DOCUMENT_TYPE],
                "real_personal_data": False,
                "visible_watermark": "SYNTHETIC / NOT VALID",
                "portraits": "geometric_avatars_only",
                "identifier_policy": (
                    "deterministic_fictional_format_examples_not_authority_issued"
                ),
            },
            "format_assurance": {
                document_type: DOCUMENT_SPECS[document_type].assurance
                for document_type in POSITIVE_DOCUMENT_TYPES
            },
            "quality_gates": {
                "identity_leakage": "pass",
                "base_split_preservation": "pass",
                "full_image_decode": "pass",
                "decoded_images": audit.loaded_images,
                "unique_synthetic_identifiers_per_type": "pass",
                "aadhaar_verhoeff": "pass",
                "all_target_classes_present": "pass",
                "hospital_hotel_hard_negatives_present": "pass",
            },
        }
    )
    _write_json_atomic(output / "build_report.json", report)
    readme = output / "README.md"
    readme.write_text(
        "# Identity foundation v2\n\n"
        "This manifest combines the approved v1 mock datasets with local, visibly "
        "watermarked, fictional synthetic identity documents and hospital/hotel hard "
        "negatives. Base splits are preserved; generated identities never cross "
        "splits.\n\n"
        "Synthetic identifiers are training examples, not authority-issued identities. "
        "Only Aadhaar examples have an implemented checksum gate. Candidate-pattern "
        "classes remain explicitly labelled in `build_report.json`.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build India-priority, privacy-safe identity foundation v2"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--registry", type=Path, default=Path("datasets/manifests/registry.yaml")
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("datasets/processed/id-foundation-v1/records.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/processed/id-foundation-v2")
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=Path("datasets/generated/synthetic-south-asia-id-v2"),
    )
    parser.add_argument("--seed", default="pii-id-foundation-v2-20261010")
    parser.add_argument("--india-identities-per-type", type=int, default=120)
    parser.add_argument("--regional-identities-per-type", type=int, default=60)
    parser.add_argument("--negative-identities", type=int, default=360)
    args = parser.parse_args()

    root = args.root.resolve(strict=True)
    report = build_foundation_v2(
        root=root,
        registry=root / args.registry
        if not args.registry.is_absolute()
        else args.registry,
        base_manifest=(
            root / args.base_manifest
            if not args.base_manifest.is_absolute()
            else args.base_manifest
        ),
        output=root / args.output if not args.output.is_absolute() else args.output,
        generated_root=(
            root / args.generated_root
            if not args.generated_root.is_absolute()
            else args.generated_root
        ),
        seed=args.seed,
        india_identities_per_type=args.india_identities_per_type,
        regional_identities_per_type=args.regional_identities_per_type,
        negative_identities=args.negative_identities,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": report["record_count"],
                "identities": report["identity_count"],
                "quality_gates": report["quality_gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
