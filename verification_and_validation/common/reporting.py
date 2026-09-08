"""Creation and strict validation of per-variant verification JSON reports."""

import json
import math
from pathlib import Path


REQUIRED_CHECK_FIELDS = (
    "name",
    "oracle",
    "observed",
    "expected",
    "error",
    "tolerance",
    "units",
    "passed",
)


def make_check(name, observed, expected, error, tolerance, units, oracle):
    check = {
        "name": str(name),
        "oracle": dict(oracle),
        "observed": float(observed),
        "expected": float(expected),
        "error": float(error),
        "tolerance": float(tolerance),
        "units": str(units),
        "passed": bool(float(error) <= float(tolerance)),
    }
    validate_verification_report({"checks": [check]})
    return check


def build_verification_report(case_id, checks, diagnostics=None, artifacts=None, schema_version=1):
    checks = list(checks)
    report = {
        "schema_version": int(schema_version),
        "case": str(case_id),
        "passed": all(check.get("passed") is True for check in checks),
        "checks": checks,
        "diagnostics": dict(diagnostics or {}),
        "artifacts": {key: str(value) for key, value in (artifacts or {}).items()},
    }
    return validate_verification_report(report)


def _validate_check(check, index, names):
    if not isinstance(check, dict):
        raise ValueError(f"oracle check {index} is not an object")
    missing = [field for field in REQUIRED_CHECK_FIELDS if field not in check]
    if missing:
        raise ValueError(f"oracle check {index} is missing: {', '.join(missing)}")
    if not isinstance(check["name"], str) or not check["name"].strip():
        raise ValueError(f"oracle check {index} has an invalid name")
    if check["name"] in names:
        raise ValueError(f"oracle check names must be unique: {check['name']}")
    names.add(check["name"])
    oracle = check["oracle"]
    if not isinstance(oracle, dict) or not isinstance(oracle.get("type"), str) or not oracle["type"].strip():
        raise ValueError(f"oracle check {index} requires oracle provenance with a nonempty type")
    if not isinstance(check["units"], str) or not check["units"].strip():
        raise ValueError(f"oracle check {index} requires nonempty units")
    for field in ("observed", "expected", "error", "tolerance"):
        value = check[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"oracle check {index} has a non-finite numeric '{field}' value")
    if check["error"] < 0 or check["tolerance"] < 0:
        raise ValueError(f"oracle check {index} requires non-negative error and tolerance")
    if not isinstance(check["passed"], bool):
        raise ValueError(f"oracle check {index} has a non-boolean 'passed' value")
    if check["passed"] != (check["error"] <= check["tolerance"]):
        raise ValueError(f"oracle check {index} pass flag is inconsistent with error and tolerance")


def validate_verification_report(report, tolerances=None):
    if not isinstance(report, dict):
        raise ValueError("oracle report must contain a JSON object")
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("oracle report must contain at least one comparison in 'checks'")
    names = set()
    for index, check in enumerate(checks):
        _validate_check(check, index, names)
    if "passed" in report and report["passed"] != all(check["passed"] for check in checks):
        raise ValueError("oracle report pass flag is inconsistent with its checks")
    for field in ("diagnostics", "artifacts"):
        if field in report and not isinstance(report[field], dict):
            raise ValueError(f"oracle report '{field}' must be a mapping")
    if tolerances is not None:
        validate_report_tolerances(report, tolerances)
    return report


def validate_report_tolerances(report, tolerances):
    checks = {check["name"]: check for check in report["checks"]}
    missing = set(tolerances).difference(checks)
    unexpected = set(checks).difference(tolerances)
    if missing:
        raise ValueError("oracle report is missing declared tolerance check(s): " + ", ".join(sorted(missing)))
    if unexpected:
        raise ValueError("oracle report has undeclared tolerance check(s): " + ", ".join(sorted(unexpected)))
    for name, tolerance in tolerances.items():
        reported = checks[name]["tolerance"]
        if float(reported) != float(tolerance):
            raise ValueError(
                f"oracle check '{name}' reports tolerance {reported}, but the resolved manifest declares {tolerance}"
            )


def read_verification_report(path, tolerances=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing oracle report: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid oracle report JSON: {error}") from error
    return validate_verification_report(report, tolerances=tolerances)


def write_verification_report(path, report, tolerances=None):
    path = Path(path)
    validate_verification_report(report, tolerances=tolerances)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
