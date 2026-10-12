#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pii_id.integration.readiness import assess_readiness  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the local ML bundle against its DetectionGodX pin"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/document/foundation-v2-best.pt"),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/inference/mixed-identities.yaml"),
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("artifacts/reports/foundation-v2-validation.json"),
    )
    parser.add_argument(
        "--integration-config",
        type=Path,
        default=Path("configs/integration/detectiongodx-v2.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    report = assess_readiness(
        root=ROOT,
        checkpoint_path=arguments.checkpoint,
        policy_path=arguments.policy,
        validation_report_path=arguments.validation_report,
        integration_config_path=arguments.integration_config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "candidate_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
