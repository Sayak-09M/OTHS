from __future__ import annotations

import csv
import io
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

SCRIPT_LANGUAGES = {
    "assamese": "asm",
    "bengali": "ben",
    "devanagari": "hin",
    "gujarati": "guj",
    "gurmukhi": "pan",
    "kannada": "kan",
    "malayalam": "mal",
    "nepali": "nep",
    "odia": "ori",
    "tamil": "tam",
    "telugu": "tel",
    "sinhala": "sin",
    "latin": "eng",
    "latin_or_cyrillic": "eng",
}
SCRIPT_PRIORITY = tuple(SCRIPT_LANGUAGES)


class OcrEngineError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class OcrToken:
    text: str
    confidence: float
    polygon: tuple[tuple[int, int], ...]
    line_id: tuple[int, int, int, int]
    source: str = "tesseract-lstm"


def languages_for_scripts(scripts: Sequence[str]) -> tuple[str, ...]:
    unknown = set(scripts) - set(SCRIPT_LANGUAGES)
    if unknown:
        raise ValueError("unsupported OCR script")
    requested = set(scripts)
    languages = [
        SCRIPT_LANGUAGES[script]
        for script in SCRIPT_PRIORITY
        if script in requested and SCRIPT_LANGUAGES[script] != "eng"
    ]
    languages.append("eng")
    return tuple(dict.fromkeys(languages))


def parse_tesseract_tsv(tsv: str) -> tuple[OcrToken, ...]:
    tokens: list[OcrToken] = []
    # Tesseract TSV is tab-delimited but is not CSV-quoted. OCR can legitimately
    # emit a lone double quote; enabling CSV quoting then merges subsequent rows.
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    required = {
        "level",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise OcrEngineError("invalid_tsv_header")
    try:
        for row in reader:
            text = (row["text"] or "").strip()
            confidence = float(row["conf"])
            if row["level"] != "5" or not text or confidence < 0:
                continue
            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
            tokens.append(
                OcrToken(
                    text=text,
                    confidence=max(0.0, min(1.0, confidence / 100.0)),
                    polygon=(
                        (left, top),
                        (left + width, top),
                        (left + width, top + height),
                        (left, top + height),
                    ),
                    line_id=(
                        int(row["page_num"]),
                        int(row["block_num"]),
                        int(row["par_num"]),
                        int(row["line_num"]),
                    ),
                )
            )
    except (KeyError, TypeError, ValueError) as error:
        raise OcrEngineError("invalid_tsv_row") from error
    return tuple(tokens)


class TesseractOcr:
    """Bounded local Tesseract LSTM adapter with fixed, allowlisted arguments."""

    def __init__(
        self,
        *,
        executable: Path,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if timeout_seconds <= 0 or max_output_bytes <= 0:
            raise ValueError("OCR resource limits must be positive")
        resolved = executable.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("OCR executable must be a regular file")
        self.executable = resolved
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def recognize(
        self, image: Image.Image, *, scripts: Sequence[str]
    ) -> tuple[OcrToken, ...]:
        languages = languages_for_scripts(scripts)
        encoded = io.BytesIO()
        image.convert("RGB").save(encoded, format="PNG")
        command = [
            str(self.executable),
            "stdin",
            "stdout",
            "--oem",
            "1",
            "--psm",
            "6",
            "-l",
            "+".join(languages),
            "tsv",
        ]
        try:
            result = subprocess.run(
                command,
                input=encoded.getvalue(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise OcrEngineError("ocr_timeout") from error
        except OSError as error:
            raise OcrEngineError("ocr_unavailable") from error
        if result.returncode != 0:
            raise OcrEngineError("ocr_failed")
        if len(result.stdout) > self.max_output_bytes:
            raise OcrEngineError("ocr_output_limit_exceeded")
        try:
            tsv = result.stdout.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise OcrEngineError("ocr_output_not_utf8") from error
        return parse_tesseract_tsv(tsv)
