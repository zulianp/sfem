#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SUITE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SUITE_DIR.parent


def bootstrap_python() -> None:
    repository_venv = ROOT_DIR / "venv"
    repository_python = ROOT_DIR / "venv" / "bin" / "python"
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


def load_case(path, yaml_module):
    with path.open(encoding="utf-8") as stream:
        data = yaml_module.safe_load(stream)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version: 1")
    for key in ("id", "name", "mesh", "driver", "verification"):
        if key not in data:
            raise ValueError(f"{path}: missing required key '{key}'")
    if not isinstance(data["id"], str) or not data["id"] or Path(data["id"]).name != data["id"]:
        raise ValueError(f"{path}: id must be a non-empty path component")
    return data


def validate_oracle_report(path):
    if not path.is_file():
        raise ValueError(f"missing oracle report: {path}")
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("oracle report must contain at least one comparison")
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise ValueError(f"oracle check {index} is not an object")
        for key in ("name", "oracle", "observed", "expected", "error", "tolerance", "passed"):
            if key not in check:
                raise ValueError(f"oracle check {index} is missing '{key}'")
        if not isinstance(check["passed"], bool):
            raise ValueError(f"oracle check {index} has a non-boolean 'passed' value")
    return report


def run_case(case_path, case, build_dir, output_root, verbose):
    case_id = case["id"]
    case_dir = case_path.parent.resolve()
    resolved_output_root = output_root.resolve()
    case_output = (resolved_output_root / case_id).resolve()
    if case_output.parent != resolved_output_root:
        raise ValueError(f"refusing unsafe case output path: {case_output}")
    if case_output.exists():
        shutil.rmtree(case_output)
    case_output.mkdir(parents=True)

    variables = {
        "root": str(ROOT_DIR.resolve()),
        "suite_dir": str(SUITE_DIR.resolve()),
        "case_dir": str(case_dir),
        "output": str(case_output),
        "mesh": str(case_output / "mesh"),
        "build_dir": str(build_dir.resolve()),
        "python": str(ROOT_DIR / "venv" / "bin" / "python"),
    }
    log_path = case_output / "run.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT_DIR / "python")
    durations = {}

    mesh_command = expand(case["mesh"]["command"], variables)
    code, _, durations["mesh"] = run_stage("mesh", mesh_command, environment, log_path, verbose)
    if code:
        raise RuntimeError(f"mesh generator exited with status {code}")

    for input_spec in case.get("inputs", []):
        template_path = case_dir / input_spec["template"]
        output_path = Path(expand(input_spec["output"], variables))
        rendered = template_path.read_text(encoding="utf-8")
        for key, value in variables.items():
            rendered = rendered.replace("{" + key + "}", value)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")

    driver = case["driver"]
    executable = expand(driver["executable"], variables)
    if not Path(executable).is_file():
        raise RuntimeError(f"missing driver: {executable}; build the requested target first")
    driver_environment = environment.copy()
    driver_environment.update({key: str(value) for key, value in expand(driver.get("environment", {}), variables).items()})
    driver_command = [executable, *expand(driver.get("arguments", []), variables)]
    code, driver_output, durations["driver"] = run_stage(
        "driver", driver_command, driver_environment, log_path, verbose
    )
    if code:
        raise RuntimeError(f"driver exited with status {code}")

    forbidden = case["verification"].get("forbidden_output", [])
    matches = [pattern for pattern in forbidden if pattern in driver_output]
    if matches:
        raise RuntimeError("driver reported non-convergence: " + ", ".join(matches))

    for requirement in case["verification"].get("required_output", []):
        pattern = requirement["pattern"]
        expected_count = int(requirement["count"])
        observed_count = driver_output.count(pattern)
        if observed_count != expected_count:
            raise RuntimeError(
                f"driver output contained {observed_count} occurrences of {pattern!r}; expected {expected_count}"
            )

    verification = case["verification"]
    verify_command = expand(verification["command"], variables)
    code, _, durations["verification"] = run_stage(
        "verification", verify_command, environment, log_path, verbose
    )
    if code:
        raise RuntimeError(f"oracle postprocessor exited with status {code}")

    oracle_report_path = Path(expand(verification["report"], variables))
    oracle_report = validate_oracle_report(oracle_report_path)
    checks = oracle_report["checks"]
    passed = all(check["passed"] for check in checks)
    failed_names = [check["name"] for check in checks if not check["passed"]]
    return {
        "id": case_id,
        "name": case["name"],
        "status": "PASS" if passed else "FAIL",
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


def print_summary(results, report_path):
    print("\nSFEM verification and validation")
    print("=" * 72)
    print(f"{'STATUS':<8} {'CASE':<34} {'CHECKS':>10} {'TIME':>10}")
    print("-" * 72)
    for result in results:
        checks = f"{result.get('checks_passed', 0)}/{result.get('checks_total', 0)}"
        print(f"{result['status']:<8} {result['id']:<34} {checks:>10} {result['duration_seconds']:>9.2f}s")
        detail = result.get("error")
        if not detail and result.get("failed_checks"):
            detail = "outside tolerance: " + ", ".join(result["failed_checks"])
        if detail:
            print(f"         {detail}")
    passed = sum(result["status"] == "PASS" for result in results)
    print("-" * 72)
    print(f"Result: {passed} passed, {len(results) - passed} failed, {len(results)} total")
    print(f"Report: {report_path}")


def main():
    bootstrap_python()
    import yaml

    parser = argparse.ArgumentParser(description="Run SFEM verification and validation cases")
    parser.add_argument("--case", action="append", dest="cases", help="run only this case id; may be repeated")
    parser.add_argument("--build-dir", type=Path, default=ROOT_DIR / "build64")
    parser.add_argument("--output-dir", type=Path, default=SUITE_DIR / "output")
    parser.add_argument("--list", action="store_true", help="list discovered cases without running them")
    parser.add_argument("--verbose", action="store_true", help="show driver and postprocessor output")
    args = parser.parse_args()

    discovered = []
    for case_path in sorted(SUITE_DIR.glob("*/case.yaml")):
        case = load_case(case_path, yaml)
        discovered.append((case_path, case))
    if args.cases:
        selected = set(args.cases)
        discovered = [item for item in discovered if item[1]["id"] in selected]
        missing = selected.difference(item[1]["id"] for item in discovered)
        if missing:
            parser.error("unknown case(s): " + ", ".join(sorted(missing)))
    if args.list:
        for _, case in discovered:
            print(f"{case['id']}: {case['name']}")
        return 0
    if not discovered:
        parser.error("no verification cases discovered")

    output_root = args.output_dir.resolve()
    if output_root in (ROOT_DIR.resolve(), SUITE_DIR.resolve()):
        parser.error(f"refusing unsafe output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for case_path, case in discovered:
        print(f"Running {case['id']} ...", flush=True)
        start = time.monotonic()
        try:
            result = run_case(case_path, case, args.build_dir, output_root, args.verbose)
        except Exception as error:
            result = {
                "id": case["id"],
                "name": case["name"],
                "status": "FAIL",
                "duration_seconds": time.monotonic() - start,
                "checks_passed": 0,
                "checks_total": 0,
                "error": str(error),
                "log": str(output_root / case["id"] / "run.log"),
            }
        results.append(result)

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT_DIR.resolve()),
        "build_directory": str(args.build_dir.resolve()),
        "status": "PASS" if all(result["status"] == "PASS" for result in results) else "FAIL",
        "cases": results,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(results, report_path)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
