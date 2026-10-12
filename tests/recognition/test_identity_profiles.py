from __future__ import annotations

import unittest

from pii_id.recognition.identity_profiles import (
    IDENTITY_PROFILES,
    mrz_check_digit,
    profile_for,
    validate_identity_value,
)
from pii_id.recognition.validators import aadhaar_check_digit


class IdentityProfileTests(unittest.TestCase):
    def test_registry_covers_every_enabled_south_asian_identity(self) -> None:
        self.assertEqual(
            set(IDENTITY_PROFILES),
            {
                "IN_AADHAAR",
                "IN_PAN",
                "IN_EPIC",
                "IN_ABHA",
                "IN_PASSPORT",
                "IN_DRIVING_LICENCE",
                "NP_NATIONAL_ID",
                "BD_NID",
                "LK_NIC",
            },
        )

    def test_each_profile_owns_its_normalizer_and_validator(self) -> None:
        self.assertEqual(
            profile_for("IN_ABHA").normalize("12-3456-7890-1234"),
            "12345678901234",
        )
        self.assertEqual(profile_for("IN_PAN").normalize("abcpd 1234 f"), "ABCPD1234F")
        with self.assertRaisesRegex(ValueError, "unsupported identity type"):
            profile_for("UNKNOWN_ID")

    def test_aadhaar_uses_verhoeff_as_strong_validation(self) -> None:
        base = "23456789012"
        value = base + aadhaar_check_digit(base)

        valid = validate_identity_value("IN_AADHAAR", value)
        invalid = validate_identity_value("IN_AADHAAR", value[:-1] + "0")

        self.assertTrue(valid.format_valid)
        self.assertTrue(valid.checksum_valid)
        self.assertIn("AADHAAR_VERHOEFF_VALID", valid.signal_codes)
        self.assertFalse(invalid.checksum_valid)
        self.assertIn("AADHAAR_VERHOEFF_INVALID", invalid.signal_codes)

    def test_pan_holder_type_is_validated_by_its_own_profile(self) -> None:
        valid = validate_identity_value("IN_PAN", "ABCPD1234F")
        invalid = validate_identity_value("IN_PAN", "ABCZD1234F")

        self.assertTrue(valid.format_valid)
        self.assertIn("PAN_HOLDER_TYPE_VALID", valid.signal_codes)
        self.assertFalse(invalid.format_valid)
        self.assertIn("PAN_HOLDER_TYPE_INVALID", invalid.signal_codes)

    def test_passport_mrz_check_digit_is_used_when_mrz_is_present(self) -> None:
        passport_number = "A1234567"
        mrz_field = passport_number + "<"
        check_digit = mrz_check_digit(mrz_field)
        lines = (
            "P<INDDOE<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            f"{mrz_field}{check_digit}IND9001011F3001012<<<<<<<<<<<<<<00",
        )

        valid = validate_identity_value(
            "IN_PASSPORT", passport_number, document_lines=lines
        )
        invalid_lines = (
            lines[0],
            lines[1][:9]
            + str((int(check_digit) + 1) % 10)
            + lines[1][10:],
        )
        invalid = validate_identity_value(
            "IN_PASSPORT", passport_number, document_lines=invalid_lines
        )

        self.assertTrue(valid.checksum_valid)
        self.assertIn("PASSPORT_MRZ_CHECK_VALID", valid.signal_codes)
        self.assertFalse(invalid.checksum_valid)
        self.assertIn("PASSPORT_MRZ_CHECK_INVALID", invalid.signal_codes)

    def test_sri_lanka_nic_rejects_impossible_embedded_day(self) -> None:
        valid = validate_identity_value("LK_NIC", "855420159V")
        invalid = validate_identity_value("LK_NIC", "859990159V")

        self.assertTrue(valid.format_valid)
        self.assertIn("LK_NIC_EMBEDDED_DAY_VALID", valid.signal_codes)
        self.assertFalse(invalid.format_valid)
        self.assertIn("LK_NIC_EMBEDDED_DAY_INVALID", invalid.signal_codes)


if __name__ == "__main__":
    unittest.main()
