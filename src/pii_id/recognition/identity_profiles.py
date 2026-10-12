from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .validators import normalize_decimal_digits, validate_aadhaar


@dataclass(frozen=True)
class IdentityValidation:
    format_valid: bool
    checksum_valid: bool | None
    signal_codes: tuple[str, ...]


Validator = Callable[[str, Sequence[str]], IdentityValidation]


@dataclass(frozen=True)
class IdentityProfile:
    id_type: str
    country: str
    discovery_pattern: re.Pattern[str]
    keywords: tuple[str, ...]
    validator: Validator
    digits_only: bool = False
    context_required_for_discovery: bool = False
    visual_class: str | None = None

    def normalize(self, value: str) -> str:
        if self.digits_only:
            return normalize_decimal_digits(value)
        return re.sub(r"[\s./-]", "", value).upper()

    def validate(
        self, value: str, *, document_lines: Sequence[str] = ()
    ) -> IdentityValidation:
        return self.validator(self.normalize(value), document_lines)


def _result(
    format_valid: bool,
    *codes: str,
    checksum_valid: bool | None = None,
) -> IdentityValidation:
    return IdentityValidation(format_valid, checksum_valid, tuple(codes))


def _aadhaar(value: str, _: Sequence[str]) -> IdentityValidation:
    syntax_valid = (
        len(value) == 12
        and value.isascii()
        and value.isdigit()
        and value[0] not in {"0", "1"}
        and len(set(value)) > 1
    )
    if not syntax_valid:
        return _result(False, "AADHAAR_FORMAT_INVALID", checksum_valid=False)
    checksum_valid = validate_aadhaar(value)
    return _result(
        True,
        "AADHAAR_FORMAT_VALID",
        (
            "AADHAAR_VERHOEFF_VALID"
            if checksum_valid
            else "AADHAAR_VERHOEFF_INVALID"
        ),
        checksum_valid=checksum_valid,
    )


_PAN_NORMALIZED = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")


def _pan(value: str, _: Sequence[str]) -> IdentityValidation:
    syntax_valid = _PAN_NORMALIZED.fullmatch(value) is not None
    holder_valid = syntax_valid and value[3] in "PCHABGJLFT"
    if not syntax_valid:
        return _result(False, "PAN_FORMAT_INVALID")
    if not holder_valid:
        return _result(False, "PAN_FORMAT_VALID", "PAN_HOLDER_TYPE_INVALID")
    return _result(True, "PAN_FORMAT_VALID", "PAN_HOLDER_TYPE_VALID")


def _format_only(
    pattern: re.Pattern[str], valid_code: str, invalid_code: str
) -> Validator:
    def validate(value: str, _: Sequence[str]) -> IdentityValidation:
        valid = pattern.fullmatch(value) is not None
        return _result(valid, valid_code if valid else invalid_code)

    return validate


_MRZ_VALUES = {
    **{str(value): value for value in range(10)},
    **{chr(ord("A") + value): value + 10 for value in range(26)},
    "<": 0,
}
_MRZ_WEIGHTS = (7, 3, 1)


def mrz_check_digit(value: str) -> str:
    """Return the ICAO 9303 check digit for one machine-readable field."""

    normalized = value.upper()
    if not normalized or any(character not in _MRZ_VALUES for character in normalized):
        raise ValueError("MRZ fields may contain only A-Z, 0-9, and <")
    total = sum(
        _MRZ_VALUES[character] * _MRZ_WEIGHTS[index % len(_MRZ_WEIGHTS)]
        for index, character in enumerate(normalized)
    )
    return str(total % 10)


_PASSPORT_NORMALIZED = re.compile(r"[A-Z][0-9]{7}")


def _passport(value: str, document_lines: Sequence[str]) -> IdentityValidation:
    if _PASSPORT_NORMALIZED.fullmatch(value) is None:
        return _result(False, "PASSPORT_FORMAT_INVALID")

    compact_lines = tuple(
        re.sub(r"\s+", "", line).upper() for line in document_lines if line.strip()
    )
    has_indian_mrz_header = any(line.startswith("P<IND") for line in compact_lines)
    if has_indian_mrz_header:
        field = value + "<"
        for line in compact_lines:
            start = line.find(field)
            if start < 0 or len(line) <= start + len(field):
                continue
            check_character = line[start + len(field)]
            if not check_character.isdigit():
                continue
            checksum_valid = mrz_check_digit(field) == check_character
            return _result(
                True,
                "PASSPORT_FORMAT_VALID",
                (
                    "PASSPORT_MRZ_CHECK_VALID"
                    if checksum_valid
                    else "PASSPORT_MRZ_CHECK_INVALID"
                ),
                checksum_valid=checksum_valid,
            )
    return _result(True, "PASSPORT_FORMAT_VALID", "PASSPORT_MRZ_UNAVAILABLE")


_LK_NIC_NORMALIZED = re.compile(r"(?:[0-9]{9}[VX]|(?:19|20)[0-9]{10})")


def _sri_lanka_nic(value: str, _: Sequence[str]) -> IdentityValidation:
    if _LK_NIC_NORMALIZED.fullmatch(value) is None:
        return _result(False, "LK_NIC_FORMAT_INVALID")
    encoded_day = int(value[2:5] if value[-1] in "VX" else value[4:7])
    calendar_day = encoded_day - 500 if encoded_day > 500 else encoded_day
    if not 1 <= calendar_day <= 366:
        return _result(
            False, "LK_NIC_FORMAT_VALID", "LK_NIC_EMBEDDED_DAY_INVALID"
        )
    return _result(True, "LK_NIC_FORMAT_VALID", "LK_NIC_EMBEDDED_DAY_VALID")


_DIGIT = r"\d"
_SEPARATOR = r"[\s-]?"

IDENTITY_PROFILES: dict[str, IdentityProfile] = {
    profile.id_type: profile
    for profile in (
        IdentityProfile(
            id_type="IN_AADHAAR",
            country="IN",
            discovery_pattern=re.compile(
                rf"(?<!\d)(?:{_DIGIT}{_SEPARATOR}){{11}}{_DIGIT}(?!\d)"
            ),
            keywords=(
                "aadhaar",
                "aadhar",
                "uidai",
                "आधार",
                "আধার",
                "ஆதார்",
                "ఆధార్",
            ),
            validator=_aadhaar,
            digits_only=True,
        ),
        IdentityProfile(
            id_type="IN_PAN",
            country="IN",
            discovery_pattern=re.compile(
                r"(?<![A-Z0-9])[A-Z]{5}\s?[0-9]{4}\s?[A-Z](?![A-Z0-9])"
            ),
            keywords=(
                "permanent account number",
                "income tax",
                "pan card",
                " pan ",
                "स्थायी खाता संख्या",
                "आयकर विभाग",
            ),
            validator=_pan,
        ),
        IdentityProfile(
            id_type="IN_EPIC",
            country="IN",
            discovery_pattern=re.compile(
                r"(?<![A-Z0-9])[A-Z]{3}\s?[0-9]{7}(?![A-Z0-9])"
            ),
            keywords=(
                "epic",
                "voter id",
                "elector",
                "election commission",
                "मतदाता पहचान पत्र",
                "निर्वाचन आयोग",
                "ভোটার পরিচয়পত্র",
            ),
            validator=_format_only(
                re.compile(r"[A-Z]{3}[0-9]{7}"),
                "EPIC_FORMAT_VALID",
                "EPIC_FORMAT_INVALID",
            ),
        ),
        IdentityProfile(
            id_type="IN_ABHA",
            country="IN",
            discovery_pattern=re.compile(
                rf"(?<!\d)(?:{_DIGIT}{_SEPARATOR}){{13}}{_DIGIT}(?!\d)"
            ),
            keywords=(
                "abha",
                "ayushman bharat health account",
                "health id",
                "आभा",
                "स्वास्थ्य खाता",
            ),
            validator=_format_only(
                re.compile(r"[0-9]{14}"),
                "ABHA_FORMAT_VALID",
                "ABHA_FORMAT_INVALID",
            ),
            digits_only=True,
            context_required_for_discovery=True,
        ),
        IdentityProfile(
            id_type="IN_PASSPORT",
            country="IN",
            discovery_pattern=re.compile(
                r"(?<![A-Z0-9])[A-Z][0-9]{7}(?![A-Z0-9])"
            ),
            keywords=(
                "passport no",
                "passport number",
                "republic of india",
                "p<ind",
                "पासपोर्ट",
                "পাসপোর্ট",
            ),
            validator=_passport,
            context_required_for_discovery=True,
        ),
        IdentityProfile(
            id_type="IN_DRIVING_LICENCE",
            country="IN",
            discovery_pattern=re.compile(
                r"(?<![A-Z0-9])[A-Z]{2}[ \t./-]*[0-9]{2}[ \t./-]*[0-9]{4}"
                r"[ \t./-]*[0-9]{7}(?![A-Z0-9])"
            ),
            keywords=(
                "driving licence",
                "driving license",
                "dl no",
                "sarathi",
                "parivahan",
                "ड्राइविंग लाइसेंस",
                "चालक अनुज्ञप्ति",
                "ড্রাইভিং লাইসেন্স",
            ),
            validator=_format_only(
                re.compile(r"[A-Z]{2}[0-9]{13}"),
                "DRIVING_LICENCE_FORMAT_VALID",
                "DRIVING_LICENCE_FORMAT_INVALID",
            ),
            context_required_for_discovery=True,
        ),
        IdentityProfile(
            id_type="NP_NATIONAL_ID",
            country="NP",
            discovery_pattern=re.compile(r"(?<!\d)\d{10}(?!\d)"),
            keywords=(
                "nepal national identity",
                "national id nepal",
                "राष्ट्रिय परिचय",
                "राष्ट्रिय परिचयपत्र",
            ),
            validator=_format_only(
                re.compile(r"[0-9]{10}"),
                "NP_NATIONAL_ID_FORMAT_VALID",
                "NP_NATIONAL_ID_FORMAT_INVALID",
            ),
            digits_only=True,
            context_required_for_discovery=True,
            visual_class="NP_ID_CARD",
        ),
        IdentityProfile(
            id_type="BD_NID",
            country="BD",
            discovery_pattern=re.compile(
                r"(?<!\d)(?:\d{10}|\d{13}|\d{17})(?!\d)"
            ),
            keywords=(
                "bangladesh nid",
                "national identity card bangladesh",
                "জাতীয় পরিচয়",
                "জাতীয় পরিচয়পত্র",
            ),
            validator=_format_only(
                re.compile(r"(?:[0-9]{10}|[0-9]{13}|[0-9]{17})"),
                "BD_NID_FORMAT_VALID",
                "BD_NID_FORMAT_INVALID",
            ),
            digits_only=True,
            context_required_for_discovery=True,
        ),
        IdentityProfile(
            id_type="LK_NIC",
            country="LK",
            discovery_pattern=re.compile(
                r"(?<![A-Z0-9])(?:\d{9}[VX]|(?:19|20)\d{10})(?![A-Z0-9])"
            ),
            keywords=(
                "sri lanka nic",
                "national identity card sri lanka",
                "ජාතික හැඳුනුම්පත",
                "தேசிய அடையாள அட்டை",
            ),
            validator=_sri_lanka_nic,
            context_required_for_discovery=True,
        ),
    )
}


def profile_for(id_type: str) -> IdentityProfile:
    try:
        return IDENTITY_PROFILES[id_type]
    except KeyError as error:
        raise ValueError(f"unsupported identity type: {id_type}") from error


def validate_identity_value(
    id_type: str,
    value: str,
    *,
    document_lines: Sequence[str] = (),
) -> IdentityValidation:
    return profile_for(id_type).validate(value, document_lines=document_lines)
