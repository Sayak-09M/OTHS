from __future__ import annotations

import io
import struct
import tempfile
import unittest
import warnings
import zipfile
import zlib
from pathlib import Path

from PIL import Image

from pii_id.inference.media import (
    ImageUnit,
    MediaError,
    MediaLimits,
    TextUnit,
    iter_media,
)


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 12), "white").save(output, format="PNG")
    return output.getvalue()


def _mat_element(data_type: int, payload: bytes) -> bytes:
    padding = b"\x00" * ((-len(payload)) % 8)
    return struct.pack("<II", data_type, len(payload)) + payload + padding


def _mat_char_matrix(name: str, text: str) -> bytes:
    flags = _mat_element(6, struct.pack("<II", 4, 0))
    dimensions = _mat_element(5, struct.pack("<ii", 1, len(text)))
    array_name = _mat_element(1, name.encode("ascii"))
    characters = _mat_element(4, text.encode("utf-16le"))
    return _mat_element(14, flags + dimensions + array_name + characters)


def _mat_double_matrix(name: str, *values: float) -> bytes:
    flags = _mat_element(6, struct.pack("<II", 6, 0))
    dimensions = _mat_element(5, struct.pack("<ii", 1, len(values)))
    array_name = _mat_element(1, name.encode("ascii"))
    numbers = _mat_element(9, struct.pack(f"<{len(values)}d", *values))
    return _mat_element(14, flags + dimensions + array_name + numbers)


def _mat_opaque_matrix(text: str) -> bytes:
    flags = _mat_element(6, struct.pack("<II", 17, 0))
    metadata = (
        _mat_element(1, b"")
        + _mat_element(1, b"MCOS")
        + _mat_element(1, b"opaque")
    )
    return _mat_element(14, flags + metadata + _mat_char_matrix("", text))


def _mat_v5(*matrices: bytes, compressed: bool = False) -> bytes:
    body = b"".join(matrices)
    if compressed:
        compressed_body = zlib.compress(body)
        body = struct.pack("<II", 15, len(compressed_body)) + compressed_body
    header = b"MATLAB 5.0 MAT-file test fixture".ljust(116, b" ")
    return header + (b"\x00" * 8) + struct.pack("<H", 0x0100) + b"IM" + body


class MediaTests(unittest.TestCase):
    def test_directory_and_zip_images_are_streamed_with_locations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plain.png").write_bytes(_png())
            with zipfile.ZipFile(root / "batch.zip", "w") as archive:
                archive.writestr("nested/card.png", _png())

            items = list(iter_media(root))

        images = [item for item in items if isinstance(item, ImageUnit)]
        self.assertEqual(len(images), 2)
        self.assertTrue(
            any("batch.zip::nested/card.png" in item.location for item in images)
        )

    def test_corrupt_image_is_an_error_not_an_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.png"
            path.write_bytes(b"not an image")
            items = list(iter_media(path))

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], MediaError)
        self.assertEqual(items[0].code, "corrupt_or_unsupported_image")

    def test_jpeg_exif_orientation_is_applied_before_model_and_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            image = Image.new("RGB", (10, 20), "white")
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, format="JPEG", exif=exif)

            items = list(iter_media(path))

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], ImageUnit)
        self.assertEqual(items[0].image.size, (20, 10))

    def test_plain_text_and_office_open_xml_are_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "records.txt").write_text(
                "Permanent Account Number PAN ABCPD1234F", encoding="utf-8"
            )
            with zipfile.ZipFile(root / "records.docx", "w") as archive:
                archive.writestr(
                    "word/document.xml",
                    '<w:document xmlns:w="urn:test"><w:t>Aadhaar</w:t>'
                    "<w:t>2345 6789 0123</w:t></w:document>",
                )

            items = list(iter_media(root))

        texts = [item.text for item in items if isinstance(item, TextUnit)]
        self.assertEqual(len(texts), 2)
        self.assertTrue(any("ABCPD1234F" in text for text in texts))
        self.assertTrue(any("Aadhaar" in text for text in texts))

    def test_matlab_source_is_extracted_as_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.m"
            path.write_text(
                "% Permanent Account Number PAN ABCPD1234F", encoding="utf-8"
            )

            items = list(iter_media(path))

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], TextUnit)
        self.assertIn("ABCPD1234F", items[0].text)

    def test_matlab_v5_char_array_is_extracted_without_executing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.mat"
            path.write_bytes(
                _mat_v5(
                    _mat_char_matrix(
                        "identity", "Permanent Account Number PAN ABCPD1234F"
                    ),
                    compressed=True,
                )
            )

            items = list(iter_media(path))

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], TextUnit)
        self.assertIn("identity", items[0].text)
        self.assertIn("ABCPD1234F", items[0].text)

    def test_corrupt_matlab_file_is_logged_as_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.mat"
            path.write_bytes(b"not a MATLAB file")

            items = list(iter_media(path))

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], MediaError)
        self.assertEqual(items[0].code, "corrupt_or_unsupported_matlab")

    def test_matlab_numeric_arrays_require_identity_related_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.mat"
            path.write_bytes(
                _mat_v5(
                    _mat_double_matrix("signal", 2_345_678_901_234),
                    _mat_double_matrix("aadhaar_number", 2_345_678_901_234),
                )
            )

            items = list(iter_media(path))

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], TextUnit)
        self.assertIn("aadhaar_number\n2345678901234", items[0].text)
        self.assertNotIn("signal\n2345678901234", items[0].text)

    def test_matlab_compressed_content_obeys_decompression_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.mat"
            path.write_bytes(
                _mat_v5(_mat_char_matrix("note", "x" * 1_024), compressed=True)
            )

            items = list(
                iter_media(path, MediaLimits(max_matlab_decompressed_bytes=64))
            )

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], MediaError)
        self.assertEqual(items[0].code, "matlab_resource_limit_exceeded")

    def test_matlab_v73_is_an_explicit_backend_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hdf5.mat"
            path.write_bytes(b"MATLAB 7.3 MAT-file".ljust(128, b" "))

            items = list(iter_media(path))

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], MediaError)
        self.assertEqual(items[0].code, "matlab_v73_backend_unavailable")

    def test_matlab_figure_opaque_container_is_traversed_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chart.fig"
            path.write_bytes(
                _mat_v5(
                    _mat_opaque_matrix(
                        "Permanent Account Number PAN ABCPD1234F"
                    )
                )
            )

            items = list(iter_media(path))

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], TextUnit)
        self.assertIn("ABCPD1234F", items[0].text)

    def test_unknown_text_file_is_detected_by_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.unknown"
            path.write_bytes(b"Permanent Account Number PAN ABCPD1234F")

            items = list(iter_media(path))

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], TextUnit)
        self.assertIn("ABCPD1234F", items[0].text)

    def test_unknown_binary_file_is_logged_instead_of_silently_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.unknown"
            path.write_bytes(b"\x00\xff\x10\x81binary\x00")

            items = list(iter_media(path))

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], MediaError)
        self.assertEqual(items[0].code, "unsupported_file_type")

    def test_unknown_text_archive_member_is_detected_by_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "legacy.record", "Permanent Account Number PAN ABCPD1234F"
                )

            items = list(iter_media(path))

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], TextUnit)
        self.assertIn("ABCPD1234F", items[0].text)

    def test_byte_identical_files_at_different_paths_are_both_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first").mkdir()
            (root / "second").mkdir()
            payload = _png()
            (root / "first" / "card.png").write_bytes(payload)
            (root / "second" / "card-copy.png").write_bytes(payload)

            items = list(iter_media(root))

        images = [item for item in items if isinstance(item, ImageUnit)]
        self.assertEqual(len(images), 2)
        self.assertEqual(len({item.location for item in images}), 2)

    def test_duplicate_zip_member_names_get_distinct_scan_locations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "duplicates.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("card.png", _png())
                    archive.writestr("card.png", _png())

            items = list(iter_media(archive_path))

        images = [item for item in items if isinstance(item, ImageUnit)]
        self.assertEqual(len(images), 2)
        self.assertEqual(len({item.location for item in images}), 2)


if __name__ == "__main__":
    unittest.main()
