import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))
import run_all  # noqa: E402


def valid_v2_case():
    return {
        "schema_version": 2,
        "id": "linear_patch_2d",
        "name": "Two-dimensional linear patch",
        "kind": "verification",
        "family": "elastic",
        "dimension": 2,
        "tier": "fast",
        "source": {
            "description": "Closed-form affine solution",
            "reference": "oracle/README.md",
        },
        "oracle": {
            "type": "analytical",
            "implementation": "oracle.py",
        },
        "mesh": {
            "command": ["{python}", "{case_dir}/generate_mesh.py", "{mesh}"],
        },
        "inputs": [],
        "verification": {
            "command": ["{python}", "{case_dir}/verify.py", "{resolved_case}"],
            "report": "{output}/verification.json",
            "tolerances": {
                "displacement_relative_l2": 1.0e-10,
                "reaction_relative": 1.0e-9,
            },
        },
        "variants": [
            {
                "id": "generated_quad4",
                "operator": "GeneratedLinearElasticity",
                "element": "QUAD4",
                "resolution": {"nx": 8, "ny": 8},
                "driver": {
                    "executable": "{build_dir}/linear_elasticity",
                    "arguments": ["{mesh}", "{output}/dirichlet.yaml", "{output}/solution"],
                    "environment": {"SFEM_OPERATOR": "{operator}"},
                },
                "expected_output": {
                    "required": [{"pattern": "converged", "count": 1}],
                    "forbidden": ["No progress made"],
                },
                "tolerances": {
                    "displacement_relative_l2": 1.0e-11,
                },
            }
        ],
    }


class ManifestTests(unittest.TestCase):
    def test_existing_v1_case_normalizes_to_default_variant(self):
        path = SUITE_DIR / "cylindrical_pressure_vessel" / "case.yaml"
        case = run_all.load_case(path, yaml)
        variants = run_all.normalized_variants(case)

        self.assertEqual(1, case["schema_version"])
        self.assertEqual(1, len(variants))
        self.assertEqual("default", variants[0]["id"])
        self.assertEqual("hyperelastic", variants[0]["family"])
        self.assertEqual(2, variants[0]["dimension"])
        self.assertEqual("GeneratedModifiedMooneyRivlin", variants[0]["operator"])
        self.assertEqual("QUAD4", variants[0]["element"])

    def test_v2_tolerance_override_and_resolution_variables(self):
        case = valid_v2_case()
        run_all.validate_case(case)
        variant = run_all.normalized_variants(case)[0]

        self.assertEqual(1.0e-11, variant["tolerances"]["displacement_relative_l2"])
        self.assertEqual(1.0e-9, variant["tolerances"]["reaction_relative"])
        self.assertEqual(
            {"resolution": '{"nx":8,"ny":8}', "resolution_nx": "8", "resolution_ny": "8"},
            run_all._resolution_variables(variant["resolution"]),
        )

    def test_material_parameter_map_is_normalized_per_variant(self):
        case = valid_v2_case()
        case["material"] = {"mu": 2.0, "lambda": 3.0}
        case["variants"][0]["material_parameter_map"] = {"mu": "mu", "lambda": "lmbda"}
        run_all.validate_case(case)
        variant = run_all.normalized_variants(case)[0]

        self.assertEqual({"mu": 2.0, "lambda": 3.0}, variant["material"])
        self.assertEqual("lmbda", variant["material_parameter_map"]["lambda"])

    def test_material_parameter_map_rejects_non_identifier_values(self):
        case = valid_v2_case()
        case["variants"][0]["material_parameter_map"] = {"lambda": "not a key"}
        with self.assertRaisesRegex(run_all.ManifestError, "expected a non-empty string|must contain only"):
            run_all.validate_case(case)

    def test_v2_rejects_duplicate_variant_ids(self):
        case = valid_v2_case()
        case["variants"].append(copy.deepcopy(case["variants"][0]))
        with self.assertRaisesRegex(run_all.ManifestError, "duplicate variant id"):
            run_all.validate_case(case)

    def test_v2_rejects_missing_tolerance_overrides(self):
        case = valid_v2_case()
        del case["variants"][0]["tolerances"]
        with self.assertRaisesRegex(run_all.ManifestError, "tolerances: missing required key"):
            run_all.validate_case(case)

    def test_v2_rejects_missing_oracle_provenance(self):
        case = valid_v2_case()
        case["source"] = {"description": "Closed-form affine solution"}
        with self.assertRaisesRegex(run_all.ManifestError, "oracle provenance"):
            run_all.validate_case(case)

    def test_discovery_validates_every_manifest_before_execution(self):
        invalid = valid_v2_case()
        del invalid["verification"]["tolerances"]
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = Path(temp_dir)
            case_dir = suite / invalid["id"]
            case_dir.mkdir()
            (case_dir / "case.yaml").write_text(yaml.safe_dump(invalid), encoding="utf-8")
            with mock.patch.object(run_all, "run_stage") as run_stage:
                with self.assertRaisesRegex(run_all.ManifestError, "tolerances"):
                    run_all.discover_cases(suite, yaml)
                run_stage.assert_not_called()


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.case = valid_v2_case()
        second = copy.deepcopy(self.case["variants"][0])
        second.update(
            {
                "id": "legacy_tri3",
                "operator": "LinearElasticity",
                "element": "TRI3",
                "tier": "medium",
            }
        )
        self.case["variants"].append(second)
        run_all.validate_case(self.case)
        self.discovered = [
            {
                "path": Path("linear_patch_2d/case.yaml"),
                "case": self.case,
                "variants": run_all.normalized_variants(self.case),
            }
        ]

    def test_filters_are_combined_and_repeated_values_are_alternatives(self):
        selected = run_all.select_cases(
            self.discovered,
            {
                "families": {"elastic"},
                "dimensions": {2},
                "tiers": {"fast", "medium"},
                "operators": {"GeneratedLinearElasticity"},
            },
        )
        self.assertEqual(["generated_quad4"], [item["id"] for item in selected[0]["variants"]])

    def test_qualified_variant_selector(self):
        selected = run_all.select_cases(
            self.discovered,
            {"variants": {"linear_patch_2d/legacy_tri3"}},
        )
        self.assertEqual(["legacy_tri3"], [item["id"] for item in selected[0]["variants"]])

    def test_unknown_variant_selector_is_rejected(self):
        with self.assertRaisesRegex(run_all.ManifestError, "unknown variant selector"):
            run_all.validate_filter_values(self.discovered, {"variants": {"missing"}})


class ResultTests(unittest.TestCase):
    def test_suite_report_is_yaml_and_preserves_scalar_types(self):
        report = {
            "schema_version": 2,
            "status": "PASS",
            "coverage": {"variants": {"covered": 1}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            report_path = run_all.write_suite_report(output, report, yaml)
            loaded = yaml.safe_load(report_path.read_text(encoding="utf-8"))

            self.assertEqual(output / "report.yaml", report_path)
            self.assertFalse((output / "report.json").exists())
            self.assertEqual(report, loaded)

    def test_error_precedes_failure_and_skip(self):
        self.assertEqual("ERROR", run_all._aggregate_status(["PASS", "FAIL", "ERROR", "SKIP"]))
        self.assertEqual("FAIL", run_all._aggregate_status(["PASS", "FAIL", "SKIP"]))
        self.assertEqual("PASS", run_all._aggregate_status(["PASS", "SKIP"]))
        self.assertEqual("SKIP", run_all._aggregate_status(["SKIP"]))

    def test_skip_is_not_counted_as_coverage(self):
        case = valid_v2_case()
        variants = run_all.normalized_variants(case)
        passed = {**run_all._result_metadata(case, variants[0]), "status": "PASS", "covered": True,
                  "duration_seconds": 0.1, "checks_passed": 1, "checks_total": 1}
        skipped_variant = copy.deepcopy(variants[0])
        skipped_variant["id"] = "optional_device"
        skipped = {**run_all._result_metadata(case, skipped_variant), "status": "SKIP", "covered": False,
                   "duration_seconds": 0.0, "checks_passed": 0, "checks_total": 0, "skip_reason": "no device"}
        case_result = run_all._case_result(case, [passed, skipped])
        coverage = run_all.coverage_summary(
            [{"case": case, "variants": [variants[0], skipped_variant]}],
            [case_result],
        )

        self.assertEqual("PASS", case_result["status"])
        self.assertFalse(case_result["covered"])
        self.assertEqual(1, coverage["variants"]["covered"])
        self.assertEqual(1, coverage["variants"]["skip"])
        self.assertEqual(0, coverage["cases"]["covered"])

    def test_oracle_report_rejects_inconsistent_pass_flag(self):
        report = {
            "checks": [
                {
                    "name": "physical_error",
                    "oracle": {"type": "analytical"},
                    "observed": 2.0,
                    "expected": 0.0,
                    "error": 2.0,
                    "tolerance": 1.0,
                    "units": "1",
                    "passed": True,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "verification.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pass flag is inconsistent"):
                run_all.validate_oracle_report(path)

    def test_oracle_report_rejects_empty_checks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "verification.json"
            path.write_text('{"checks": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one comparison"):
                run_all.validate_oracle_report(path)

    def test_report_tolerances_must_match_manifest(self):
        report = {
            "checks": [
                {
                    "name": "physical_error",
                    "oracle": {"type": "analytical"},
                    "observed": 0.0,
                    "expected": 0.0,
                    "error": 0.0,
                    "tolerance": 2.0,
                    "units": "1",
                    "passed": True,
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "resolved manifest declares"):
            run_all.validate_report_tolerances(report, {"physical_error": 1.0})

    def test_run_variant_writes_resolved_manifest_and_reports_pass(self):
        case = valid_v2_case()
        case["material"] = {"mu": 2.0, "lambda": 3.0}
        case["variants"][0]["material_parameter_map"] = {"mu": "mu", "lambda": "lmbda"}
        case["inputs"] = [{"template": "operator.yaml", "output": "{output}/operator.yaml"}]
        case["verification"]["tolerances"] = {"displacement_relative_l2": 1.0e-10}
        true_executable = shutil.which("true")
        self.assertIsNotNone(true_executable)
        case["mesh"]["command"] = [true_executable]
        case["verification"]["command"] = [true_executable]
        case["variants"][0]["driver"] = {
            "executable": true_executable,
            "arguments": [],
            "environment": {},
        }
        run_all.validate_case(case)
        variant = run_all.normalized_variants(case)[0]

        def fake_stage(name, command, environment, log_file, verbose):
            log_file.touch(exist_ok=True)
            if name == "verification":
                report = {
                    "checks": [
                        {
                            "name": "displacement_relative_l2",
                            "oracle": {"type": "analytical"},
                            "observed": 0.0,
                            "expected": 0.0,
                            "error": 0.0,
                            "tolerance": 1.0e-11,
                            "units": "1",
                            "passed": True,
                        }
                    ]
                }
                (log_file.parent / "verification.json").write_text(json.dumps(report), encoding="utf-8")
            output = "converged\n" if name == "driver" else ""
            return 0, output, 0.01

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_dir = root / case["id"]
            case_dir.mkdir()
            case_path = case_dir / "case.yaml"
            case_path.write_text(yaml.safe_dump(case), encoding="utf-8")
            (case_dir / "operator.yaml").write_text(
                "operator:\n  type: {operator}\n  {material_key_lambda}: {material_lambda}\n",
                encoding="utf-8",
            )
            output = root / "output"
            with mock.patch.object(run_all, "run_stage", side_effect=fake_stage):
                result = run_all.run_variant(case_path, case, variant, root / "build", output, False)

            self.assertEqual("PASS", result["status"])
            resolved = yaml.safe_load((output / "resolved_case.yaml").read_text(encoding="utf-8"))
            self.assertEqual("generated_quad4", resolved["selected_variant"]["id"])
            self.assertEqual(1.0e-11, resolved["verification"]["tolerances"]["displacement_relative_l2"])
            rendered_operator = yaml.safe_load((output / "operator.yaml").read_text(encoding="utf-8"))
            self.assertEqual(3.0, rendered_operator["operator"]["lmbda"])


if __name__ == "__main__":
    unittest.main()
