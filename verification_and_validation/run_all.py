#!/usr/bin/env python3

import argparse
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SUITE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SUITE_DIR.parent
SCHEMA_VERSIONS = (1, 2)
KINDS = ("verification", "validation")
TIERS = ("fast", "medium", "extended")
STATUS_ORDER = ("PASS", "FAIL", "ERROR", "SKIP")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
RESOLUTION_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ManifestError(ValueError):
    pass


def bootstrap_python() -> None:
    repository_venv = ROOT_DIR / "venv"
    repository_python = repository_venv / "bin" / "python"
    if repository_python.exists() and Path(sys.prefix).resolve() != repository_venv.resolve():
        os.execv(str(repository_python), [str(repository_python), str(Path(__file__).resolve()), *sys.argv[1:]])


def expand(value, variables):
    if isinstance(value, str):
        return value.format_map(variables)
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    return value


def run_stage(name, command, environment, log_file, verbose):
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration = time.monotonic() - start
    with log_file.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{name}] {' '.join(command)}\n")
        stream.write(completed.stdout)
        if completed.stdout and not completed.stdout.endswith("\n"):
            stream.write("\n")
        stream.write(f"[{name}] exit={completed.returncode} duration={duration:.3f}s\n")
    if verbose and completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    return completed.returncode, completed.stdout, duration


def _error(path, message):
    raise ManifestError(f"{path}: {message}")


def _require_mapping(value, path, nonempty=False):
    if not isinstance(value, dict) or (nonempty and not value):
        _error(path, "expected a non-empty mapping" if nonempty else "expected a mapping")
    return value


def _require_list(value, path, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        _error(path, "expected a non-empty list" if nonempty else "expected a list")
    return value


def _require_string(value, path):
    if not isinstance(value, str) or not value.strip():
        _error(path, "expected a non-empty string")
    return value


def _require_identifier(value, path):
    _require_string(value, path)
    if not IDENTIFIER_RE.fullmatch(value):
        _error(path, "must contain only letters, digits, '.', '_', and '-' and must start with a letter or digit")
    return value


def _validate_command(value, path):
    command = _require_list(value, path, nonempty=True)
    for index, item in enumerate(command):
        _require_string(item, f"{path}[{index}]")


def _validate_environment(value, path):
    environment = _require_mapping(value, path)
    for key, item in environment.items():
        _require_string(key, f"{path} key")
        if isinstance(item, (dict, list)) or item is None:
            _error(f"{path}.{key}", "expected a scalar value")


def _validate_material(value, path):
    material = _require_mapping(value, path)
    for key, item in material.items():
        _require_identifier(key, f"{path} key")
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            _error(f"{path}.{key}", "expected a finite number")


def _validate_material_parameter_map(value, path):
    mapping = _require_mapping(value, path)
    for physical_name, operator_name in mapping.items():
        _require_identifier(physical_name, f"{path} key")
        _require_identifier(operator_name, f"{path}.{physical_name}")


def _validate_tolerances(value, path, nonempty=True):
    tolerances = _require_mapping(value, path, nonempty=nonempty)
    for key, tolerance in tolerances.items():
        _require_string(key, f"{path} key")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            _error(f"{path}.{key}", "expected a finite non-negative number")
        if not math.isfinite(float(tolerance)) or tolerance < 0:
            _error(f"{path}.{key}", "expected a finite non-negative number")


def _validate_required_output(value, path):
    requirements = _require_list(value, path)
    for index, requirement in enumerate(requirements):
        item_path = f"{path}[{index}]"
        requirement = _require_mapping(requirement, item_path)
        _require_string(requirement.get("pattern"), f"{item_path}.pattern")
        count = requirement.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _error(f"{item_path}.count", "expected a non-negative integer")


def _validate_forbidden_output(value, path):
    patterns = _require_list(value, path)
    for index, pattern in enumerate(patterns):
        _require_string(pattern, f"{path}[{index}]")


def _validate_inputs(value, path):
    inputs = _require_list(value, path)
    for index, input_spec in enumerate(inputs):
        item_path = f"{path}[{index}]"
        input_spec = _require_mapping(input_spec, item_path)
        _require_string(input_spec.get("template"), f"{item_path}.template")
        _require_string(input_spec.get("output"), f"{item_path}.output")


def _validate_mesh(value, path):
    mesh = _require_mapping(value, path, nonempty=True)
    _validate_command(mesh.get("command"), f"{path}.command")


def _validate_driver(value, path, require_environment=False):
    driver = _require_mapping(value, path, nonempty=True)
    _require_string(driver.get("executable"), f"{path}.executable")
    arguments = driver.get("arguments", [])
    _require_list(arguments, f"{path}.arguments")
    for index, argument in enumerate(arguments):
        _require_string(argument, f"{path}.arguments[{index}]")
    if require_environment and "environment" not in driver:
        _error(f"{path}.environment", "missing required key")
    _validate_environment(driver.get("environment", {}), f"{path}.environment")


def _validate_verification(value, path, require_tolerances=True):
    verification = _require_mapping(value, path, nonempty=True)
    _validate_command(verification.get("command"), f"{path}.command")
    _require_string(verification.get("report"), f"{path}.report")
    if require_tolerances and "tolerances" not in verification:
        _error(f"{path}.tolerances", "missing required key")
    _validate_tolerances(verification.get("tolerances", {}), f"{path}.tolerances", nonempty=require_tolerances)


def _validate_provenance(data, path, require_oracle_type):
    source = _require_mapping(data.get("source"), f"{path}.source", nonempty=True)
    _require_string(source.get("description"), f"{path}.source.description")
    references = [value for key, value in source.items() if key != "description" and value not in (None, "", [], {})]
    if not references:
        _error(f"{path}.source", "oracle provenance requires at least one reference in addition to description")
    oracle = _require_mapping(data.get("oracle"), f"{path}.oracle", nonempty=True)
    if require_oracle_type:
        _require_string(oracle.get("type"), f"{path}.oracle.type")


def _validate_metadata(data, path, required):
    for key in ("id", "name"):
        if key not in data:
            _error(f"{path}.{key}", "missing required key")
    _require_identifier(data["id"], f"{path}.id")
    _require_string(data["name"], f"{path}.name")

    for key in ("family", "tier", "kind"):
        if required and key not in data:
            _error(f"{path}.{key}", "missing required key")
        if key in data:
            _require_identifier(data[key], f"{path}.{key}")

    if "kind" in data and data["kind"] not in KINDS:
        _error(f"{path}.kind", f"expected one of {', '.join(KINDS)}")
    if "tier" in data and data["tier"] not in TIERS:
        _error(f"{path}.tier", f"expected one of {', '.join(TIERS)}")

    if required and "dimension" not in data:
        _error(f"{path}.dimension", "missing required key")
    if "dimension" in data:
        dimension = data["dimension"]
        if isinstance(dimension, bool) or dimension not in (2, 3):
            _error(f"{path}.dimension", "expected 2 or 3")


def _validate_resolution(value, path):
    if isinstance(value, bool) or value is None or isinstance(value, (list, tuple)):
        _error(path, "expected a scalar or non-empty mapping")
    if isinstance(value, dict):
        if not value:
            _error(path, "expected a non-empty mapping")
        for key, item in value.items():
            if not isinstance(key, str) or not RESOLUTION_KEY_RE.fullmatch(key):
                _error(f"{path} key", "expected an identifier usable as a template variable")
            if isinstance(item, bool) or isinstance(item, (dict, list)) or item is None:
                _error(f"{path}.{key}", "expected a scalar value")
            if isinstance(item, str) and not item:
                _error(f"{path}.{key}", "expected a non-empty scalar value")
            if isinstance(item, (int, float)) and not math.isfinite(float(item)):
                _error(f"{path}.{key}", "expected a finite scalar value")
    elif not isinstance(value, (str, int, float)):
        _error(path, "expected a scalar or non-empty mapping")
    elif isinstance(value, str) and not value:
        _error(path, "expected a non-empty scalar value")
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        _error(path, "expected a finite scalar value")


def _validate_v1(data, path):
    _validate_metadata(data, path, required=False)
    for key in ("mesh", "driver", "verification"):
        if key not in data:
            _error(f"{path}.{key}", "missing required key")
    _validate_provenance(data, path, require_oracle_type=False)
    _validate_mesh(data["mesh"], f"{path}.mesh")
    _validate_driver(data["driver"], f"{path}.driver")
    _validate_verification(data["verification"], f"{path}.verification")
    _validate_inputs(data.get("inputs", []), f"{path}.inputs")
    _validate_forbidden_output(data["verification"].get("forbidden_output", []),
                               f"{path}.verification.forbidden_output")
    _validate_required_output(data["verification"].get("required_output", []),
                              f"{path}.verification.required_output")
    if "operator" in data:
        _require_string(data["operator"], f"{path}.operator")
    if "element" in data:
        _require_string(data["element"], f"{path}.element")
    if "resolution" in data:
        _validate_resolution(data["resolution"], f"{path}.resolution")


def _validate_v2(data, path):
    _validate_metadata(data, path, required=True)
    _validate_provenance(data, path, require_oracle_type=True)
    _validate_mesh(data.get("mesh"), f"{path}.mesh")
    _validate_inputs(data.get("inputs", []), f"{path}.inputs")
    _validate_verification(data.get("verification"), f"{path}.verification")
    _validate_material(data.get("material", {}), f"{path}.material")
    _validate_material_parameter_map(data.get("material_parameter_map", {}), f"{path}.material_parameter_map")

    variants = _require_list(data.get("variants"), f"{path}.variants", nonempty=True)
    variant_ids = set()
    for index, variant in enumerate(variants):
        variant_path = f"{path}.variants[{index}]"
        variant = _require_mapping(variant, variant_path, nonempty=True)
        for key in ("id", "operator", "element", "resolution", "driver", "expected_output", "tolerances"):
            if key not in variant:
                _error(f"{variant_path}.{key}", "missing required key")
        variant_id = _require_identifier(variant["id"], f"{variant_path}.id")
        if variant_id in variant_ids:
            _error(f"{variant_path}.id", f"duplicate variant id '{variant_id}'")
        variant_ids.add(variant_id)

        _require_string(variant["operator"], f"{variant_path}.operator")
        _require_string(variant["element"], f"{variant_path}.element")
        _validate_resolution(variant["resolution"], f"{variant_path}.resolution")
        _validate_driver(variant["driver"], f"{variant_path}.driver", require_environment=True)
        _validate_tolerances(variant["tolerances"], f"{variant_path}.tolerances")

        expected = _require_mapping(variant["expected_output"], f"{variant_path}.expected_output")
        for key in ("required", "forbidden"):
            if key not in expected:
                _error(f"{variant_path}.expected_output.{key}", "missing required key")
        _validate_required_output(expected["required"], f"{variant_path}.expected_output.required")
        _validate_forbidden_output(expected["forbidden"], f"{variant_path}.expected_output.forbidden")

        if "mesh" in variant:
            _validate_mesh(variant["mesh"], f"{variant_path}.mesh")
        if "inputs" in variant:
            _validate_inputs(variant["inputs"], f"{variant_path}.inputs")
        if "verification" in variant:
            _validate_verification(variant["verification"], f"{variant_path}.verification", require_tolerances=False)
        if "material" in variant:
            _validate_material(variant["material"], f"{variant_path}.material")
        if "material_parameter_map" in variant:
            _validate_material_parameter_map(
                variant["material_parameter_map"], f"{variant_path}.material_parameter_map"
            )

        for key in ("family", "tier"):
            if key in variant:
                _require_identifier(variant[key], f"{variant_path}.{key}")
        if "tier" in variant and variant["tier"] not in TIERS:
            _error(f"{variant_path}.tier", f"expected one of {', '.join(TIERS)}")
        if "dimension" in variant and (isinstance(variant["dimension"], bool) or variant["dimension"] not in (2, 3)):
            _error(f"{variant_path}.dimension", "expected 2 or 3")

        if "skip" in variant:
            skip = _require_mapping(variant["skip"], f"{variant_path}.skip", nonempty=True)
            _require_string(skip.get("reason"), f"{variant_path}.skip.reason")


def validate_case(data, path="case.yaml"):
    if not isinstance(data, dict):
        _error(path, "expected a mapping at the document root")
    version = data.get("schema_version")
    if version not in SCHEMA_VERSIONS:
        _error(f"{path}.schema_version", f"expected one of {SCHEMA_VERSIONS}")
    if version == 1:
        _validate_v1(data, path)
    else:
        _validate_v2(data, path)
    return data


def load_case(path, yaml_module):
    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml_module.safe_load(stream)
    except Exception as error:
        raise ManifestError(f"{path}: unable to parse YAML: {error}") from error
    return validate_case(data, str(path))


def _oracle_type(case):
    oracle = case.get("oracle", {})
    return oracle.get("type", case.get("kind", "unspecified"))


def _operator_from_v1(case):
    if case.get("operator"):
        return case["operator"]
    return case.get("driver", {}).get("environment", {}).get("SFEM_OPERATOR", "unspecified")


def normalized_variants(case):
    common = {
        "family": case.get("family", "unspecified"),
        "dimension": case.get("dimension", "unspecified"),
        "tier": case.get("tier", "unspecified"),
        "oracle_type": _oracle_type(case),
    }
    if case["schema_version"] == 1:
        verification = copy.deepcopy(case["verification"])
        return [
            {
                **common,
                "id": "default",
                "operator": _operator_from_v1(case),
                "element": case.get("element", "unspecified"),
                "resolution": copy.deepcopy(case.get("resolution", "unspecified")),
                "mesh": copy.deepcopy(case["mesh"]),
                "inputs": copy.deepcopy(case.get("inputs", [])),
                "driver": copy.deepcopy(case["driver"]),
                "expected_output": {
                    "required": copy.deepcopy(verification.get("required_output", [])),
                    "forbidden": copy.deepcopy(verification.get("forbidden_output", [])),
                },
                "verification": verification,
                "tolerances": copy.deepcopy(verification["tolerances"]),
                "skip": None,
                "schema_version": 1,
            }
        ]

    variants = []
    for item in case["variants"]:
        verification = copy.deepcopy(item.get("verification", case["verification"]))
        merged_tolerances = copy.deepcopy(case["verification"]["tolerances"])
        merged_tolerances.update(item["tolerances"])
        verification["tolerances"] = merged_tolerances
        variants.append(
            {
                **common,
                "family": item.get("family", common["family"]),
                "dimension": item.get("dimension", common["dimension"]),
                "tier": item.get("tier", common["tier"]),
                "id": item["id"],
                "operator": item["operator"],
                "element": item["element"],
                "resolution": copy.deepcopy(item["resolution"]),
                "mesh": copy.deepcopy(item.get("mesh", case["mesh"])),
                "inputs": copy.deepcopy(item.get("inputs", case.get("inputs", []))),
                "driver": copy.deepcopy(item["driver"]),
                "expected_output": copy.deepcopy(item["expected_output"]),
                "verification": verification,
                "tolerances": merged_tolerances,
                "material": copy.deepcopy(item.get("material", case.get("material", {}))),
                "material_parameter_map": copy.deepcopy(
                    item.get("material_parameter_map", case.get("material_parameter_map", {}))
                ),
                "skip": copy.deepcopy(item.get("skip")),
                "schema_version": 2,
            }
        )
    return variants


def discover_cases(suite_dir, yaml_module):
    discovered = []
    case_ids = {}
    for case_path in sorted(suite_dir.glob("*/case.yaml")):
        case = load_case(case_path, yaml_module)
        case_id = case["id"]
        if case_id in case_ids:
            raise ManifestError(f"{case_path}: duplicate case id '{case_id}' also used by {case_ids[case_id]}")
        case_ids[case_id] = case_path
        discovered.append({"path": case_path, "case": case, "variants": normalized_variants(case)})
    return discovered


def _matches_variant_selector(case_id, variant_id, selectors):
    if not selectors:
        return True
    qualified = f"{case_id}/{variant_id}"
    return variant_id in selectors or qualified in selectors


def select_cases(discovered, filters):
    selected = []
    selected_case_ids = filters.get("cases")
    for entry in discovered:
        case = entry["case"]
        if selected_case_ids and case["id"] not in selected_case_ids:
            continue
        variants = []
        for variant in entry["variants"]:
            if filters.get("families") and variant["family"] not in filters["families"]:
                continue
            if filters.get("dimensions") and variant["dimension"] not in filters["dimensions"]:
                continue
            if filters.get("tiers") and variant["tier"] not in filters["tiers"]:
                continue
            if filters.get("operators") and variant["operator"] not in filters["operators"]:
                continue
            if not _matches_variant_selector(case["id"], variant["id"], filters.get("variants")):
                continue
            variants.append(variant)
        if variants:
            selected.append({**entry, "variants": variants})
    return selected


def validate_filter_values(discovered, filters):
    known_cases = {entry["case"]["id"] for entry in discovered}
    missing_cases = set(filters.get("cases") or ()).difference(known_cases)
    if missing_cases:
        raise ManifestError("unknown case(s): " + ", ".join(sorted(missing_cases)))

    selectors = set(filters.get("variants") or ())
    if selectors:
        matched = set()
        for entry in discovered:
            case_id = entry["case"]["id"]
            for variant in entry["variants"]:
                if variant["id"] in selectors:
                    matched.add(variant["id"])
                qualified = f"{case_id}/{variant['id']}"
                if qualified in selectors:
                    matched.add(qualified)
        missing = selectors.difference(matched)
        if missing:
            raise ManifestError("unknown variant selector(s): " + ", ".join(sorted(missing)))


def validate_oracle_report(path):
    from common.reporting import read_verification_report

    return read_verification_report(path)


def validate_report_tolerances(report, tolerances):
    from common.reporting import validate_report_tolerances as validate

    return validate(report, tolerances)


def _resolution_variables(resolution):
    variables = {}
    if isinstance(resolution, dict):
        variables["resolution"] = json.dumps(resolution, sort_keys=True, separators=(",", ":"))
        for key, value in resolution.items():
            variables[f"resolution_{key}"] = str(value)
    else:
        variables["resolution"] = str(resolution)
    return variables


def _result_metadata(case, variant):
    return {
        "id": variant["id"],
        "family": variant["family"],
        "dimension": variant["dimension"],
        "tier": variant["tier"],
        "operator": variant["operator"],
        "element": variant["element"],
        "resolution": variant["resolution"],
        "oracle_type": variant["oracle_type"],
        "schema_version": variant["schema_version"],
        "case_id": case["id"],
    }


def _resolved_manifest(case, variant):
    resolved = copy.deepcopy(case)
    resolved["selected_variant"] = {
        key: value
        for key, value in _result_metadata(case, variant).items()
        if key not in ("case_id", "schema_version")
    }
    resolved["mesh"] = copy.deepcopy(variant["mesh"])
    resolved["inputs"] = copy.deepcopy(variant["inputs"])
    resolved["driver"] = copy.deepcopy(variant["driver"])
    resolved["verification"] = copy.deepcopy(variant["verification"])
    resolved["material"] = copy.deepcopy(variant.get("material", {}))
    resolved["material_parameter_map"] = copy.deepcopy(variant.get("material_parameter_map", {}))
    return resolved


def _write_resolved_manifest(path, case, variant):
    import yaml

    path.write_text(yaml.safe_dump(_resolved_manifest(case, variant), sort_keys=False), encoding="utf-8")


def run_variant(case_path, case, variant, build_dir, variant_output, verbose):
    variant_output.mkdir(parents=True, exist_ok=True)
    case_dir = case_path.parent.resolve()
    variables = {
        "root": str(ROOT_DIR.resolve()),
        "suite_dir": str(SUITE_DIR.resolve()),
        "case_dir": str(case_dir),
        "case": str(case_path.resolve()),
        "output": str(variant_output),
        "mesh": str(variant_output / "mesh"),
        "build_dir": str(build_dir.resolve()),
        "python": str(ROOT_DIR / "venv" / "bin" / "python"),
        "variant": variant["id"],
        "variant_id": variant["id"],
        "operator": variant["operator"],
        "element": variant["element"],
    }
    variables.update(_resolution_variables(variant["resolution"]))
    for physical_name, value in variant.get("material", {}).items():
        variables[f"material_{physical_name}"] = str(value)
        variables[f"material_key_{physical_name}"] = variant.get("material_parameter_map", {}).get(
            physical_name, physical_name
        )

    resolved_case_path = variant_output / "resolved_case.yaml"
    variables["resolved_case"] = str(resolved_case_path)
    _write_resolved_manifest(resolved_case_path, case, variant)

    log_path = variant_output / "run.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT_DIR / "python")
    durations = {}

    mesh_command = expand(variant["mesh"]["command"], variables)
    code, _, durations["mesh"] = run_stage("mesh", mesh_command, environment, log_path, verbose)
    if code:
        raise RuntimeError(f"mesh generator exited with status {code}")

    for input_spec in variant["inputs"]:
        template_path = case_dir / input_spec["template"]
        if not template_path.is_file():
            raise RuntimeError(f"missing input template: {template_path}")
        output_path = Path(expand(input_spec["output"], variables))
        rendered = template_path.read_text(encoding="utf-8")
        for key, value in variables.items():
            rendered = rendered.replace("{" + key + "}", str(value))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")

    driver = variant["driver"]
    executable = expand(driver["executable"], variables)
    if not Path(executable).is_file():
        raise RuntimeError(f"missing driver: {executable}; build the requested target first")
    driver_environment = environment.copy()
    expanded_environment = expand(driver.get("environment", {}), variables)
    driver_environment.update({key: str(value) for key, value in expanded_environment.items()})
    driver_command = [executable, *expand(driver.get("arguments", []), variables)]
    code, driver_output, durations["driver"] = run_stage(
        "driver", driver_command, driver_environment, log_path, verbose
    )
    if code:
        raise RuntimeError(f"driver exited with status {code}")

    expected_output = variant["expected_output"]
    matches = [pattern for pattern in expected_output["forbidden"] if pattern in driver_output]
    if matches:
        raise RuntimeError("driver reported forbidden output: " + ", ".join(matches))

    for requirement in expected_output["required"]:
        pattern = requirement["pattern"]
        expected_count = requirement["count"]
        observed_count = driver_output.count(pattern)
        if observed_count != expected_count:
            raise RuntimeError(
                f"driver output contained {observed_count} occurrences of {pattern!r}; expected {expected_count}"
            )

    verification = variant["verification"]
    verify_command = expand(verification["command"], variables)
    code, _, durations["verification"] = run_stage(
        "verification", verify_command, environment, log_path, verbose
    )
    if code:
        raise RuntimeError(f"oracle postprocessor exited with status {code}")

    oracle_report_path = Path(expand(verification["report"], variables))
    oracle_report = validate_oracle_report(oracle_report_path)
    validate_report_tolerances(oracle_report, variant["tolerances"])
    checks = oracle_report["checks"]
    passed = all(check["passed"] for check in checks)
    failed_names = [check["name"] for check in checks if not check["passed"]]
    return {
        **_result_metadata(case, variant),
        "status": "PASS" if passed else "FAIL",
        "covered": passed,
        "duration_seconds": sum(durations.values()),
        "stage_durations_seconds": durations,
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "failed_checks": failed_names,
        "oracle_report": str(oracle_report_path),
        "log": str(log_path),
        "checks": checks,
        "diagnostics": oracle_report.get("diagnostics", {}),
        "artifacts": oracle_report.get("artifacts", {}),
    }


def _aggregate_status(statuses):
    if "ERROR" in statuses:
        return "ERROR"
    if "FAIL" in statuses:
        return "FAIL"
    if "PASS" in statuses:
        return "PASS"
    return "SKIP"


def _case_result(case, variant_results):
    status = _aggregate_status([result["status"] for result in variant_results])
    return {
        "id": case["id"],
        "name": case["name"],
        "schema_version": case["schema_version"],
        "kind": case.get("kind", "unspecified"),
        "family": case.get("family", "unspecified"),
        "dimension": case.get("dimension", "unspecified"),
        "tier": case.get("tier", "unspecified"),
        "status": status,
        "covered": bool(variant_results) and all(result["status"] == "PASS" for result in variant_results),
        "duration_seconds": sum(result["duration_seconds"] for result in variant_results),
        "checks_passed": sum(result.get("checks_passed", 0) for result in variant_results),
        "checks_total": sum(result.get("checks_total", 0) for result in variant_results),
        "variants": variant_results,
    }


def run_case(entry, build_dir, output_root, verbose):
    case_path = entry["path"]
    case = entry["case"]
    case_output = (output_root.resolve() / case["id"]).resolve()
    if case_output.parent != output_root.resolve():
        raise ValueError(f"refusing unsafe case output path: {case_output}")
    if case_output.exists():
        shutil.rmtree(case_output)

    variant_results = []
    for variant in entry["variants"]:
        if case["schema_version"] == 1:
            variant_output = case_output
        else:
            variant_output = (case_output / variant["id"]).resolve()
            if variant_output.parent != case_output:
                raise ValueError(f"refusing unsafe variant output path: {variant_output}")

        if variant["skip"]:
            variant_results.append(
                {
                    **_result_metadata(case, variant),
                    "status": "SKIP",
                    "covered": False,
                    "duration_seconds": 0.0,
                    "checks_passed": 0,
                    "checks_total": 0,
                    "skip_reason": variant["skip"]["reason"],
                    "log": None,
                }
            )
            continue

        print(f"Running {case['id']}/{variant['id']} ...", flush=True)
        start = time.monotonic()
        try:
            result = run_variant(case_path, case, variant, build_dir, variant_output, verbose)
        except Exception as error:
            result = {
                **_result_metadata(case, variant),
                "status": "ERROR",
                "covered": False,
                "duration_seconds": time.monotonic() - start,
                "checks_passed": 0,
                "checks_total": 0,
                "error": str(error),
                "log": str(variant_output / "run.log"),
            }
        variant_results.append(result)
    return _case_result(case, variant_results)


def _case_error_result(entry, error, duration):
    case = entry["case"]
    variants = []
    for variant in entry["variants"]:
        variants.append(
            {
                **_result_metadata(case, variant),
                "status": "ERROR",
                "covered": False,
                "duration_seconds": 0.0,
                "checks_passed": 0,
                "checks_total": 0,
                "error": str(error),
                "log": None,
            }
        )
    result = _case_result(case, variants)
    result["duration_seconds"] = duration
    return result


def _status_counts(results):
    counts = {status.lower(): 0 for status in STATUS_ORDER}
    for result in results:
        counts[result["status"].lower()] += 1
    counts["selected"] = len(results)
    counts["covered"] = sum(bool(result.get("covered")) for result in results)
    return counts


def _group_coverage(variant_results, key):
    groups = {}
    for result in variant_results:
        group = str(result[key])
        groups.setdefault(group, []).append(result)
    return {group: _status_counts(results) for group, results in sorted(groups.items())}


def coverage_summary(discovered, case_results):
    variant_results = [variant for case in case_results for variant in case["variants"]]
    return {
        "cases": {
            "discovered": len(discovered),
            **_status_counts(case_results),
        },
        "variants": {
            "discovered": sum(len(entry["variants"]) for entry in discovered),
            **_status_counts(variant_results),
        },
        "by_family": _group_coverage(variant_results, "family"),
        "by_dimension": _group_coverage(variant_results, "dimension"),
        "by_tier": _group_coverage(variant_results, "tier"),
        "by_operator": _group_coverage(variant_results, "operator"),
    }


def _format_resolution(resolution):
    if isinstance(resolution, dict):
        return ",".join(f"{key}={value}" for key, value in sorted(resolution.items()))
    return str(resolution)


def print_listing(selected):
    for entry in selected:
        case = entry["case"]
        for variant in entry["variants"]:
            skip = f" skip={variant['skip']['reason']}" if variant["skip"] else ""
            print(
                f"{case['id']}/{variant['id']}: {case['name']} "
                f"family={variant['family']} dimension={variant['dimension']} tier={variant['tier']} "
                f"operator={variant['operator']} element={variant['element']} "
                f"resolution={_format_resolution(variant['resolution'])}{skip}"
            )


def print_summary(results, report_path):
    print("\nSFEM verification and validation")
    print("=" * 84)
    print(f"{'STATUS':<8} {'CASE':<34} {'VARIANTS':>10} {'CHECKS':>10} {'TIME':>10}")
    print("-" * 84)
    for result in results:
        checks = f"{result['checks_passed']}/{result['checks_total']}"
        covered = sum(variant["status"] == "PASS" for variant in result["variants"])
        variants = f"{covered}/{len(result['variants'])}"
        print(
            f"{result['status']:<8} {result['id']:<34} {variants:>10} "
            f"{checks:>10} {result['duration_seconds']:>9.2f}s"
        )
        for variant in result["variants"]:
            detail = variant.get("error")
            if not detail and variant.get("failed_checks"):
                detail = "outside tolerance: " + ", ".join(variant["failed_checks"])
            if not detail and variant.get("skip_reason"):
                detail = "skipped: " + variant["skip_reason"]
            if detail:
                print(f"         {variant['id']}: {detail}")
    counts = _status_counts(results)
    print("-" * 84)
    print(
        "Result: "
        f"{counts['pass']} passed, {counts['fail']} failed, {counts['error']} errors, "
        f"{counts['skip']} skipped, {counts['selected']} total"
    )
    print(f"Report: {report_path}")


def _filters_from_args(args):
    return {
        "cases": set(args.cases or ()),
        "families": set(args.families or ()),
        "dimensions": set(args.dimensions or ()),
        "tiers": set(args.tiers or ()),
        "operators": set(args.operators or ()),
        "variants": set(args.variants or ()),
    }


def _selection_report(filters):
    return {key: sorted(values) for key, values in filters.items() if values}


def write_suite_report(output_root, report, yaml):
    report_path = output_root / "report.yaml"
    report_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return report_path


def build_parser():
    parser = argparse.ArgumentParser(description="Run SFEM verification and validation cases")
    parser.add_argument("--case", action="append", dest="cases", help="select this case id; may be repeated")
    parser.add_argument(
        "--family", action="append", dest="families", help="select this material family; may be repeated"
    )
    parser.add_argument("--dimension", action="append", dest="dimensions", type=int, choices=(2, 3),
                        help="select this spatial dimension; may be repeated")
    parser.add_argument("--tier", action="append", dest="tiers", choices=TIERS,
                        help="select this execution tier; may be repeated")
    parser.add_argument("--operator", action="append", dest="operators", help="select this operator; may be repeated")
    parser.add_argument("--variant", action="append", dest="variants",
                        help="select a variant id or case_id/variant_id; may be repeated")
    parser.add_argument("--build-dir", type=Path, default=ROOT_DIR / "build64")
    parser.add_argument("--output-dir", type=Path, default=SUITE_DIR / "output")
    parser.add_argument("--list", action="store_true", help="list selected case variants without running them")
    parser.add_argument("--verbose", action="store_true", help="show driver and postprocessor output")
    return parser


def main(argv=None):
    bootstrap_python()
    import yaml

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        discovered = discover_cases(SUITE_DIR, yaml)
        filters = _filters_from_args(args)
        validate_filter_values(discovered, filters)
        selected = select_cases(discovered, filters)
    except ManifestError as error:
        parser.error(str(error))

    if not discovered:
        parser.error("no verification cases discovered")
    if not selected:
        parser.error("no case variants match the requested filters")
    if args.list:
        print_listing(selected)
        return 0

    output_root = args.output_dir.resolve()
    if output_root in (ROOT_DIR.resolve(), SUITE_DIR.resolve()):
        parser.error(f"refusing unsafe output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for entry in selected:
        start = time.monotonic()
        try:
            result = run_case(entry, args.build_dir, output_root, args.verbose)
        except Exception as error:
            result = _case_error_result(entry, error, time.monotonic() - start)
        results.append(result)
    status = _aggregate_status([result["status"] for result in results])
    report = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT_DIR.resolve()),
        "build_directory": str(args.build_dir.resolve()),
        "selection": _selection_report(filters),
        "status": status,
        "coverage": coverage_summary(discovered, results),
        "cases": results,
    }
    report_path = write_suite_report(output_root, report, yaml)
    print_summary(results, report_path)
    return 1 if status in ("FAIL", "ERROR") else 0


if __name__ == "__main__":
    raise SystemExit(main())
