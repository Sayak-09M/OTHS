from __future__ import annotations

import io
import os
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

from PIL import Image, ImageOps, UnidentifiedImageError

from .matlab import MatlabExtractionError, extract_matlab_text

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TEXT_EXTENSIONS = {
    ".csv",
    ".html",
    ".htm",
    ".json",
    ".log",
    ".m",
    ".md",
    ".text",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
MATLAB_EXTENSIONS = {".fig", ".mat"}
OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
SUPPORTED_EXTENSIONS = (
    IMAGE_EXTENSIONS
    | TEXT_EXTENSIONS
    | MATLAB_EXTENSIONS
    | OFFICE_EXTENSIONS
    | {
        ".eml",
        ".pdf",
        ".zip",
    }
)


@dataclass(frozen=True)
class MediaLimits:
    max_file_bytes: int = 512 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    max_pdf_pages: int = 2_000
    max_archive_members: int = 20_000
    max_archive_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024
    max_compression_ratio: float = 200.0
    max_archive_depth: int = 2
    max_matlab_nodes: int = 200_000
    max_matlab_text_chars: int = 10_000_000
    max_matlab_decompressed_bytes: int = 128 * 1024 * 1024
    max_matlab_numeric_values: int = 4_096


@dataclass(frozen=True)
class ImageUnit:
    location: str
    image: Image.Image
    native_text: str | None = None


@dataclass(frozen=True)
class TextUnit:
    location: str
    text: str


@dataclass(frozen=True)
class MediaError:
    location: str
    code: str


MediaItem = ImageUnit | TextUnit | MediaError


def _looks_like_text(payload: bytes, *, sample_bytes: int = 64 * 1024) -> bool:
    sample = payload[:sample_bytes]
    if not sample:
        return True
    encodings: tuple[str, ...]
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ("utf-16",)
    elif b"\x00" in sample:
        encodings = ("utf-16le", "utf-16be")
    else:
        encodings = ("utf-8-sig", "windows-1252")
    for encoding in encodings:
        try:
            text = sample.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
        useful = sum(
            character.isprintable() or character in "\n\r\t\f"
            for character in text
        )
        if useful / max(1, len(text)) >= 0.95:
            return True
    return False


def _decode_image(payload: bytes, location: str, limits: MediaLimits) -> MediaItem:
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.width * source.height > limits.max_image_pixels:
                return MediaError(location, "image_pixel_limit_exceeded")
            oriented = ImageOps.exif_transpose(source)
            return ImageUnit(location, oriented.convert("RGB").copy())
    except (OSError, UnidentifiedImageError, ValueError):
        return MediaError(location, "corrupt_or_unsupported_image")


def _render_pdf(
    payload: bytes, location: str, limits: MediaLimits
) -> Iterator[MediaItem]:
    try:
        import pymupdf
    except ImportError:
        yield MediaError(location, "pdf_backend_unavailable")
        return
    try:
        with pymupdf.open(stream=payload, filetype="pdf") as document:
            if document.page_count > limits.max_pdf_pages:
                yield MediaError(location, "pdf_page_limit_exceeded")
                return
            for page_index, page in enumerate(document):
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), alpha=False)
                page_location = f"{location}#page={page_index + 1}"
                if pixmap.width * pixmap.height > limits.max_image_pixels:
                    yield MediaError(page_location, "image_pixel_limit_exceeded")
                    continue
                image = Image.frombytes(
                    "RGB", (pixmap.width, pixmap.height), pixmap.samples
                )
                native_text = page.get_text("text", sort=True)
                yield ImageUnit(page_location, image, native_text=native_text or None)
    except (RuntimeError, ValueError):
        yield MediaError(location, "corrupt_or_encrypted_pdf")


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def _decode_text(payload: bytes, location: str) -> MediaItem:
    try:
        if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = payload.decode("utf-16")
        else:
            text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = payload.decode("windows-1252")
        except UnicodeDecodeError:
            return MediaError(location, "text_decode_failed")
    return TextUnit(location, text)


def _extract_email(payload: bytes, location: str) -> MediaItem:
    try:
        message = BytesParser(policy=policy.default).parsebytes(payload)
        parts: list[str] = []
        for header in ("subject", "from", "to", "cc"):
            value = message.get(header)
            if value:
                parts.append(str(value))
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_type() not in {"text/plain", "text/html"}:
                continue
            content = part.get_content()
            if part.get_content_type() == "text/html":
                content = re.sub(r"<[^>]+>", " ", content)
            parts.append(content)
        return TextUnit(location, "\n".join(parts))
    except (KeyError, LookupError, TypeError, ValueError):
        return MediaError(location, "email_decode_failed")


def _extract_office_text(
    payload: bytes, location: str, limits: MediaLimits
) -> MediaItem:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir() and member.filename.casefold().endswith(".xml")
            ]
            if len(members) > limits.max_archive_members:
                return MediaError(location, "office_member_limit_exceeded")
            total_size = sum(member.file_size for member in members)
            if total_size > limits.max_archive_uncompressed_bytes:
                return MediaError(location, "office_size_limit_exceeded")
            text: list[str] = []
            for member in members:
                if not _safe_member(member.filename):
                    return MediaError(location, "unsafe_office_member")
                ratio = member.file_size / max(1, member.compress_size)
                if ratio > limits.max_compression_ratio:
                    return MediaError(location, "office_ratio_limit_exceeded")
                root = ElementTree.fromstring(archive.read(member))
                for element in root.iter():
                    local_name = element.tag.rsplit("}", 1)[-1]
                    if local_name in {"t", "v"} and element.text:
                        text.append(element.text)
            return TextUnit(location, "\n".join(text))
    except (ElementTree.ParseError, OSError, ValueError, zipfile.BadZipFile):
        return MediaError(location, "corrupt_office_document")


def _extract_matlab(
    payload: bytes, location: str, limits: MediaLimits
) -> MediaItem:
    try:
        text = extract_matlab_text(
            payload,
            max_nodes=limits.max_matlab_nodes,
            max_text_chars=limits.max_matlab_text_chars,
            max_decompressed_bytes=limits.max_matlab_decompressed_bytes,
            max_numeric_values=limits.max_matlab_numeric_values,
        )
    except MatlabExtractionError as error:
        return MediaError(location, error.code)
    return TextUnit(location, text)


def _read_payload(
    payload: bytes,
    *,
    suffix: str,
    location: str,
    limits: MediaLimits,
    archive_depth: int,
) -> Iterator[MediaItem]:
    if len(payload) > limits.max_file_bytes:
        yield MediaError(location, "file_size_limit_exceeded")
    elif suffix in IMAGE_EXTENSIONS:
        yield _decode_image(payload, location, limits)
    elif suffix == ".pdf":
        yield from _render_pdf(payload, location, limits)
    elif suffix in TEXT_EXTENSIONS:
        yield _decode_text(payload, location)
    elif suffix in MATLAB_EXTENSIONS:
        yield _extract_matlab(payload, location, limits)
    elif suffix in OFFICE_EXTENSIONS:
        yield _extract_office_text(payload, location, limits)
    elif suffix == ".eml":
        yield _extract_email(payload, location)
    elif suffix == ".zip":
        yield from _read_zip(payload, location, limits, archive_depth)


def _read_zip(
    payload: bytes, location: str, limits: MediaLimits, archive_depth: int
) -> Iterator[MediaItem]:
    if archive_depth >= limits.max_archive_depth:
        yield MediaError(location, "archive_depth_limit_exceeded")
        return
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) > limits.max_archive_members:
                yield MediaError(location, "archive_member_limit_exceeded")
                return
            total_size = 0
            member_occurrences: dict[str, int] = {}
            for member in members:
                if member.is_dir():
                    continue
                member_location = f"{location}::{member.filename}"
                member_occurrences[member.filename] = (
                    member_occurrences.get(member.filename, 0) + 1
                )
                occurrence = member_occurrences[member.filename]
                if occurrence > 1:
                    member_location = (
                        f"{member_location}::archive-entry={occurrence}"
                    )
                if not _safe_member(member.filename):
                    yield MediaError(member_location, "unsafe_archive_member")
                    continue
                if member.flag_bits & 0x1:
                    yield MediaError(member_location, "encrypted_archive_member")
                    continue
                total_size += member.file_size
                if total_size > limits.max_archive_uncompressed_bytes:
                    yield MediaError(location, "archive_size_limit_exceeded")
                    return
                ratio = member.file_size / max(1, member.compress_size)
                if ratio > limits.max_compression_ratio:
                    yield MediaError(member_location, "archive_ratio_limit_exceeded")
                    continue
                try:
                    member_payload = archive.read(member)
                except (OSError, RuntimeError, zipfile.BadZipFile):
                    yield MediaError(member_location, "archive_member_read_failed")
                    continue
                suffix = Path(member.filename).suffix.casefold()
                if suffix not in SUPPORTED_EXTENSIONS:
                    if _looks_like_text(member_payload):
                        yield _decode_text(member_payload, member_location)
                    else:
                        yield MediaError(
                            member_location, "unsupported_archive_member"
                        )
                    continue
                yield from _read_payload(
                    member_payload,
                    suffix=suffix,
                    location=member_location,
                    limits=limits,
                    archive_depth=archive_depth + 1,
                )
    except (OSError, ValueError, zipfile.BadZipFile):
        yield MediaError(location, "corrupt_archive")


def _walk(path: Path) -> Iterator[Path | MediaError]:
    if path.is_file():
        yield path
        return
    try:
        entries = sorted(os.scandir(path), key=lambda item: item.name.casefold())
    except (OSError, PermissionError):
        yield MediaError(str(path), "path_unreadable")
        return
    for entry in entries:
        entry_path = Path(entry.path)
        try:
            if entry.is_symlink():
                yield MediaError(str(entry_path), "symlink_skipped")
                continue
            if entry.is_dir(follow_symlinks=False):
                yield from _walk(entry_path)
            elif entry.is_file(follow_symlinks=False):
                yield entry_path
        except (OSError, PermissionError):
            yield MediaError(str(entry_path), "path_unreadable")


def iter_media(path: Path, limits: MediaLimits | None = None) -> Iterator[MediaItem]:
    limits = limits or MediaLimits()
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.exists():
        yield MediaError(str(resolved), "path_not_found")
        return
    for candidate in _walk(resolved):
        if isinstance(candidate, MediaError):
            yield candidate
            continue
        location = str(candidate)
        try:
            size = candidate.stat().st_size
            if size > limits.max_file_bytes:
                yield MediaError(location, "file_size_limit_exceeded")
                continue
            payload = candidate.read_bytes()
        except (OSError, PermissionError):
            yield MediaError(location, "file_unreadable")
            continue
        suffix = candidate.suffix.casefold()
        if suffix not in SUPPORTED_EXTENSIONS:
            if _looks_like_text(payload):
                yield _decode_text(payload, location)
            else:
                yield MediaError(location, "unsupported_file_type")
            continue
        yield from _read_payload(
            payload,
            suffix=suffix,
            location=location,
            limits=limits,
            archive_depth=0,
        )
