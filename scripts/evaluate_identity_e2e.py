#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pii_id.data.loader import DatasetLoadError, DatasetManifest  # noqa: E402
from pii_id.evaluation.access import (  # noqa: E402
    AccessPurpose,
    FrozenCandidate,
    SplitAccessGuard,
)
from pii_id.evaluation.identity_report import (  # noqa: E402
    evaluate_identity_record,
    summarize_identity_evaluation,
)
from pii_id.inference.runtime import load_scanner  # noqa: E402


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OCR, identity extraction, and policy confirmation end to end "
            "without writing complete identifiers to the report"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts/document/foundation-v2-best.pt",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "configs/inference/mixed-identities.yaml",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "calibration", "locked_test"),
        default="validation",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Override the manifest inside the project root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON report path inside the project root",
    )
    parser.add_argument(
        "--entity",
        action="append",
        dest="entities",
        help="Evaluate one enabled entity; repeat for multiple entities",
    )
    parser.add_argument("--tesseract", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Diagnostic partial run only; omitted means the complete split",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--final-evaluation",
        action="store_true",
        help="Required to open the locked test after freezing model and policy hashes",
    )
    return parser.parse_args(argv)


def _within_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_relative_to(ROOT):
        raise ValueError("path escapes project root")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    target = _within_root(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_config(checkpoint_path: Path) -> Mapping[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("format") != (
        "pii-id-document-checkpoint"
    ):
        raise ValueError("unsupported checkpoint format")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no training configuration")
    return config


def _manifest_path(
    *, options: argparse.Namespace, config: Mapping[str, object]
) -> Path:
    if options.manifest is not None:
        return _within_root(options.manifest).resolve(strict=True)
    if options.split == "validation":
        configured = config.get("validation_manifest")
        if not isinstance(configured, str):
            raise ValueError("checkpoint has no validation manifest")
        return _within_root(ROOT / configured).resolve(strict=True)
    validation = config.get("validation_manifest")
    if not isinstance(validation, str):
        raise ValueError("checkpoint has no validation manifest")
    return _within_root(ROOT / validation).parent.joinpath(
        f"{options.split}.jsonl"
    ).resolve(strict=True)


def _access_purpose(split: str, *, final_evaluation: bool) -> AccessPurpose:
    if split == "validation":
        return AccessPurpose.MODEL_SELECTION
    if split == "calibration":
        return AccessPurpose.CALIBRATION
    if not final_evaluation:
        raise ValueError(
            "locked_test requires --final-evaluation and must not be used for tuning"
        )
    return AccessPurpose.FINAL_EVALUATION


def _failed_ocr_passes(events: Sequence[object]) -> int:
    for event in events:
        if getattr(event, "code", None) != "content_extracted":
            continue
        data = getattr(event, "data", None)
        if isinstance(data, Mapping):
            value = data.get("failed_ocr_passes", 0)
            if isinstance(value, int) and value >= 0:
                return value
    return 0


def _progress(processed: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = (total - processed) / rate if rate > 0 else 0.0
    print(
        f"processed={processed}/{total} elapsed_minutes={elapsed / 60:.1f} "
        f"eta_minutes={remaining / 60:.1f}",
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    options = arguments(argv)
    if options.max_records is not None and options.max_records <= 0:
        raise ValueError("max-records must be positive")
    if options.progress_every <= 0:
        raise ValueError("progress-every must be positive")

    checkpoint_path = options.checkpoint.expanduser().resolve(strict=True)
    policy_path = options.policy.expanduser().resolve(strict=True)
    output_path = options.output or (
        ROOT / f"artifacts/reports/identity-e2e-{options.split}.json"
    )
    config = _checkpoint_config(checkpoint_path)
    manifest_path = _manifest_path(options=options, config=config)
    registry = config.get("registry")
    artifact_root = config.get("artifact_root")
    if not isinstance(registry, str) or not isinstance(artifact_root, str):
        raise ValueError("checkpoint dataset configuration is incomplete")

    purpose = _access_purpose(
        options.split, final_evaluation=options.final_evaluation
    )
    model_sha256 = _sha256(checkpoint_path)
    policy_sha256 = _sha256(policy_path)
    candidate = FrozenCandidate(
        model_sha256=model_sha256,
        config_sha256=_json_sha256(config),
        policy_sha256=policy_sha256,
    )
    loaded = load_scanner(
        checkpoint_path=checkpoint_path,
        policy_path=policy_path,
        tesseract_path=options.tesseract,
        device_name=options.device,
    )
    enabled = {
        id_type for id_type, policy in loaded.scanner.policies.items() if policy.enabled
    }
    selected = set(options.entities or enabled)
    unknown = selected - enabled
    if unknown:
        raise ValueError(f"entities are not enabled by policy: {sorted(unknown)}")

    guard = SplitAccessGuard(
        root=ROOT,
        ledger_path=_within_root(ROOT / artifact_root / "audit/dataset-access.jsonl"),
    )
    guard.authorize(
        split=options.split,
        purpose=purpose,
        run_id=f"identity-e2e-{options.split}-{model_sha256[:12]}",
        candidate=candidate if purpose is AccessPurpose.FINAL_EVALUATION else None,
    )

    manifest = DatasetManifest(
        root=ROOT,
        manifest_path=manifest_path,
        registry_path=_within_root(ROOT / registry),
    )
    records = list(manifest.iter_records())
    if any(record["split"] != options.split for record in records):
        raise ValueError("manifest contains records from a different split")
    total_available = len(records)
    if options.max_records is not None:
        records = records[: options.max_records]

    rows = []
    errors: list[dict[str, object]] = []
    started = time.monotonic()
    total = len(records)
    for index, record in enumerate(records, start=1):
        try:
            sample = manifest.load_sample(record)
            scan = loaded.scanner.scan_image_candidates(sample.image)
            event_codes = tuple(event.code for event in scan.events)
            rows.append(
                evaluate_identity_record(
                    sample.record,
                    candidates=scan.candidates,
                    selected_entities=selected,
                    rejection_reasons=loaded.scanner.candidate_rejection_reasons,
                    truth_rejection_reasons=(
                        loaded.scanner.reference_value_rejection_reasons
                    ),
                    manifest_line=index,
                    event_codes=event_codes,
                    failed_ocr_passes=_failed_ocr_passes(scan.events),
                )
            )
        except DatasetLoadError as error:
            errors.append({"error_code": error.code, "manifest_line": index})
        except (OSError, RuntimeError, TypeError, ValueError):
            errors.append(
                {
                    "error_code": "evaluation_failed",
                    "manifest_line": index,
                }
            )
        if index % options.progress_every == 0 or index == total:
            _progress(index, total, started)

    summary = summarize_identity_evaluation(rows)
    report: dict[str, object] = {
        "format": "pii-id-end-to-end-evaluation",
        "format_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "split": options.split,
        "purpose": purpose.value,
        "complete_split": options.max_records is None and not errors,
        "partial_run": options.max_records is not None,
        "total_manifest_records": total_available,
        "attempted_records": len(records),
        "successful_records": len(rows),
        "errors": errors,
        "checkpoint_sha256": model_sha256,
        "policy_sha256": policy_sha256,
        "manifest_sha256": _sha256(manifest_path),
        "metric_scope": {
            "exact": (
                "labelled id_number records plus documents outside selected identity "
                "types as negative examples"
            ),
            "weak_positive": (
                "same-type candidate presence only; excluded from exact precision and "
                "recall because the manifest has no id_number value"
            ),
            "invalid_ground_truth": (
                "included in candidate-level OCR metrics but excluded from confirmed "
                "recall when the labelled value fails the active format or checksum "
                "requirements"
            ),
            "privacy": "complete identifier values are never serialized",
            "claim_limit": (
                "dataset evaluation is not proof of production accuracy on hospital "
                "or hotel data"
            ),
        },
        **summary,
    }
    _write_json_atomic(output_path, report)
    confirmed = report["confirmed_level"]
    console = {
        "split": options.split,
        "successful_records": len(rows),
        "errors": len(errors),
        "precision": confirmed["precision"],
        "recall": confirmed["recall"],
        "f1": confirmed["f1"],
        "false_positives": confirmed["false_positives"],
        "false_negatives": confirmed["false_negatives"],
    }
    print(json.dumps(console, sort_keys=True))
    print(str(_within_root(output_path).resolve(strict=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
