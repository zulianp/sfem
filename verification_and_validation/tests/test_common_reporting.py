import json
import sys
import tempfile
import unittest
from pathlib import Path


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))

from common.reporting import (  # noqa: E402
    build_verification_report,
    make_check,
    read_verification_report,
    validate_verification_report,
    write_verification_report,
)


class ReportingTests(unittest.TestCase):
    def test_report_round_trip_and_tolerance_validation(self):
        check = make_check(
            "displacement_relative_l2",
            observed=2.0e-12,
            expected=0.0,
            error=2.0e-12,
            tolerance=1.0e-10,
            units="1",
            oracle={"type": "analytical", "reference": "oracle.py"},
        )
        report = build_verification_report(
            "synthetic",
            [check],
            diagnostics={"minimum_jacobian": 1.0},
            artifacts={"profile": Path("profile.csv")},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "verification.json"
            write_verification_report(path, report, {"displacement_relative_l2": 1.0e-10})
            loaded = read_verification_report(path, {"displacement_relative_l2": 1.0e-10})

            self.assertTrue(loaded["passed"])
            self.assertEqual("profile.csv", loaded["artifacts"]["profile"])

    def test_empty_checks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one comparison"):
            validate_verification_report({"checks": []})

    def test_nonfinite_values_are_rejected(self):
        report = {
            "checks": [
                {
                    "name": "energy",
                    "oracle": {"type": "analytical"},
                    "observed": float("nan"),
                    "expected": 0.0,
                    "error": 0.0,
                    "tolerance": 1.0,
                    "units": "J",
                    "passed": True,
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_verification_report(report)

    def test_top_level_pass_flag_must_match_checks(self):
        check = make_check("energy", 2.0, 0.0, 2.0, 1.0, "J", {"type": "analytical"})
        with self.assertRaisesRegex(ValueError, "report pass flag"):
            validate_verification_report({"passed": True, "checks": [check]})

    def test_invalid_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "verification.json"
            path.write_text(json.dumps({"checks": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one comparison"):
                read_verification_report(path)


if __name__ == "__main__":
    unittest.main()
