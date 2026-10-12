from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class AccessPurpose(StrEnum):
    FIT = "fit"
    MODEL_SELECTION = "model_selection"
    CALIBRATION = "calibration"
    FINAL_EVALUATION = "final_evaluation"
    INTEGRITY_AUDIT = "integrity_audit"


class SplitAccessError(RuntimeError):
    pass


class LockedTestAccessError(SplitAccessError):
    pass


@dataclass(frozen=True)
class FrozenCandidate:
    model_sha256: str
    config_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("candidate hashes must be lowercase SHA-256 values")


@dataclass(frozen=True)
class AccessGrant:
    split: str
    purpose: AccessPurpose
    run_id: str
    candidate: FrozenCandidate | None

    def allows(self, split: str, purpose: AccessPurpose) -> bool:
        return self.split == split and self.purpose is purpose


class SplitAccessGuard:
    _purpose_split = {
        AccessPurpose.FIT: "train",
        AccessPurpose.MODEL_SELECTION: "validation",
        AccessPurpose.CALIBRATION: "calibration",
        AccessPurpose.FINAL_EVALUATION: "locked_test",
    }

    def __init__(self, *, root: Path, ledger_path: Path) -> None:
        self.root = root.resolve(strict=True)
        resolved_parent = ledger_path.parent.resolve(strict=False)
        if not resolved_parent.is_relative_to(self.root):
            raise ValueError("ledger must be inside the configured root")
        if ledger_path.exists() and not ledger_path.resolve().is_relative_to(self.root):
            raise ValueError("ledger symlink escapes the configured root")
        self.ledger_path = ledger_path

    def _write_event(self, event: dict[str, object]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        descriptor = os.open(
            self.ledger_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, serialized.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def authorize(
        self,
        *,
        split: str,
        purpose: AccessPurpose,
        run_id: str,
        candidate: FrozenCandidate | None = None,
    ) -> AccessGrant:
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", run_id):
            raise ValueError("run_id must be a non-sensitive machine identifier")
        expected_split = self._purpose_split.get(purpose)
        allowed = purpose is AccessPurpose.INTEGRITY_AUDIT or split == expected_split
        if purpose is AccessPurpose.FINAL_EVALUATION and candidate is None:
            allowed = False
        event: dict[str, object] = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "split": split,
            "purpose": purpose.value,
            "decision": "allowed" if allowed else "denied",
        }
        if candidate is not None:
            event.update(asdict(candidate))
        self._write_event(event)
        if not allowed:
            if split == "locked_test":
                raise LockedTestAccessError("locked test access denied")
            raise SplitAccessError("split is not authorized for this purpose")
        return AccessGrant(
            split=split,
            purpose=purpose,
            run_id=run_id,
            candidate=candidate,
        )
