from __future__ import annotations

import unittest

from pii_id.recognition.ocr import languages_for_scripts, parse_tesseract_tsv


class OcrAdapterTests(unittest.TestCase):
    def test_script_mapping_is_allowlisted_and_india_first(self) -> None:
        self.assertEqual(languages_for_scripts(["devanagari"]), ("hin", "eng"))
        self.assertEqual(
            languages_for_scripts(["tamil", "devanagari"]),
            ("hin", "tam", "eng"),
        )
        with self.assertRaises(ValueError):
            languages_for_scripts(["../../unsafe"])

    def test_tsv_parser_retains_text_box_and_confidence(self) -> None:
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t96.5\tभारत\n"
            "5\t1\t1\t1\t1\t2\t42\t20\t50\t12\t88.0\t1234\n"
            "4\t1\t1\t1\t1\t0\t0\t0\t0\t0\t-1\t\n"
        )

        tokens = parse_tesseract_tsv(tsv)

        self.assertEqual([token.text for token in tokens], ["भारत", "1234"])
        self.assertEqual(tokens[0].polygon, ((10, 20), (40, 20), (40, 32), (10, 32)))
        self.assertAlmostEqual(tokens[0].confidence, 0.965)

    def test_stray_quote_does_not_swallow_following_tsv_rows(self) -> None:
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            '5\t1\t1\t1\t1\t1\t10\t20\t5\t12\t20.0\t"\n'
            "5\t1\t1\t1\t2\t1\t10\t40\t40\t12\t96.0\t7192\n"
            "5\t1\t1\t1\t2\t2\t55\t40\t40\t12\t96.0\t4940\n"
            "5\t1\t1\t1\t2\t3\t100\t40\t40\t12\t96.0\t4480\n"
        )

        tokens = parse_tesseract_tsv(tsv)

        self.assertEqual(
            [token.text for token in tokens], ['"', "7192", "4940", "4480"]
        )


if __name__ == "__main__":
    unittest.main()
