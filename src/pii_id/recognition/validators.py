from __future__ import annotations

import re
import unicodedata

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)
_PAN_PATTERN = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")


def normalize_decimal_digits(value: str) -> str:
    normalized: list[str] = []
    for character in value:
        try:
            normalized.append(str(unicodedata.decimal(character)))
        except (TypeError, ValueError):
            continue
    return "".join(normalized)


def aadhaar_check_digit(base_digits: str) -> str:
    digits = normalize_decimal_digits(base_digits)
    if len(digits) != len(base_digits) or not digits:
        raise ValueError("base must contain decimal digits only")
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[(index + 1) % 8][int(digit)]]
    return str(_VERHOEFF_INV[checksum])


def validate_aadhaar(value: str) -> bool:
    allowed = all(character.isdecimal() or character in " -" for character in value)
    digits = normalize_decimal_digits(value)
    if not allowed or len(digits) != 12 or digits[0] in {"0", "1"}:
        return False
    if len(set(digits)) == 1:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[index % 8][int(digit)]]
    return checksum == 0


def validate_pan_syntax(value: str) -> bool:
    return _PAN_PATTERN.fullmatch(value.strip().upper()) is not None


def mask_identifier(value: str, *, keep_last: int = 5) -> str:
    compact = "".join(character for character in value if not character.isspace())
    keep = min(max(0, keep_last), len(compact))
    return "*" * (len(compact) - keep) + compact[-keep:]
