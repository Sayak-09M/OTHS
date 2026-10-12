from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from pii_id.inference.media import (
    ImageUnit,
    MediaError,
    MediaLimits,
    TextUnit,
    iter_media,
)
from pii_id.inference.runtime import load_scanner


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pii-id-scan",
        description=(
            "Scan mixed images, PDFs, text, Office, MATLAB v5, email, ZIPs, "
            "or directories for supported identity PII"
        ),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/document/foundation-v2-best.pt"),
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("configs/inference/mixed-identities.yaml")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/reports/identity-scan.json")
    )
    parser.add_argument("--tesseract", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--max-pdf-pages", type=int, default=2_000)
    return parser.parse_args(argv)


def _write_report(path: Path, report: dict[str, object]) -> None:
    target = path.expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    loaded = load_scanner(
        checkpoint_path=arguments.checkpoint,
        policy_path=arguments.policy,
        tesseract_path=arguments.tesseract,
        device_name=arguments.device,
    )
    documents: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    finding_count = 0
    review_count = 0
    findings_by_type: dict[str, int] = {}
    reviews_by_type: dict[str, int] = {}
    for item in iter_media(
        arguments.input, MediaLimits(max_pdf_pages=arguments.max_pdf_pages)
    ):
        if isinstance(item, MediaError):
            errors.append(asdict(item))
            continue
        try:
            if isinstance(item, ImageUnit):
                result = loaded.scanner.scan_image(
                    item.image, native_text=item.native_text
                )
            elif isinstance(item, TextUnit):
                result = loaded.scanner.scan_text(item.text)
            else:
                raise TypeError("unsupported media item")
            findings = [asdict(finding) for finding in result.findings]
            review_candidates = [
                asdict(candidate) for candidate in result.review_candidates
            ]
            finding_count += len(findings)
            review_count += len(review_candidates)
            for finding in result.findings:
                findings_by_type[finding.id_type] = (
                    findings_by_type.get(finding.id_type, 0) + 1
                )
            for candidate in result.review_candidates:
                reviews_by_type[candidate.id_type] = (
                    reviews_by_type.get(candidate.id_type, 0) + 1
                )
            documents.append(
                {
                    "location": item.location,
                    "findings": findings,
                    "review_candidates": review_candidates,
                    "events": [event.as_log() for event in result.events],
                }
            )
        except (RuntimeError, ValueError):
            errors.append({"location": item.location, "code": "inference_failed"})
    report: dict[str, object] = {
        "schema_version": "2.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "checkpoint_metrics": dict(loaded.checkpoint_metrics),
        "summary": {
            "documents_scanned": len(documents),
            "findings": finding_count,
            "review_candidates": review_count,
            "findings_by_type": findings_by_type,
            "review_candidates_by_type": reviews_by_type,
            "errors": len(errors),
        },
        "documents": documents,
        "errors": errors,
    }
    _write_report(arguments.report, report)
    print(json.dumps(report["summary"], sort_keys=True))
    print(str(arguments.report.expanduser().resolve(strict=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
