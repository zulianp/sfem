#!/usr/bin/env python3
"""Collect CPU Newton/FGMRES counts on the public regularized cavity."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile


SUMMARY = re.compile(r"newton_converged: (\d+)\s+newton_it: (\d+) \(last stage\)\s+"
                     r"newton_total: (\d+) over (\d+) stage\(s\)\s+lin_it_total: (\d+)")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(line):
    return dict(word.split("=", 1) for word in line.split()[1:])


def parse_log(output, returncode):
    checks = []
    summary = None
    for line in output.splitlines():
        if line.startswith("linear_check: "):
            record = fields(line)
            for name in ("step", "stage", "newton", "iterations", "at_limit"):
                record[name] = int(record[name])
            for name in ("true_abs", "rhs_norm", "target"):
                value = float(record[name])
                record[name] = value if math.isfinite(value) else None
            checks.append(record)
        if line.startswith("linear_check_summary: "):
            if summary is not None:
                raise ValueError("multiple linear summaries")
            summary = fields(line)
            for name in ("checked", "failed", "dofs"):
                summary[name] = int(summary[name])
            for name in ("nonlinear_abs", "nonlinear_initial", "nonlinear_rtol", "nonlinear_atol"):
                value = float(summary[name])
                summary[name] = value if math.isfinite(value) else None
    matches = list(SUMMARY.finditer(output))
    result = {"returncode": returncode, "linear_checks": checks,
              "status": "incomplete", "driver_summary": summary}
    if len(matches) != 1 or summary is None:
        return result
    converged, last_newton, newton, stages, linear = map(int, matches[0].groups())
    result.update(newton_iterations=newton, last_stage_newton=last_newton,
                  linear_iterations=linear, stages=stages,
                  driver_converged=bool(converged))
    valid = (len(checks) == summary["checked"] and len(checks) > 0
             and sum(c["iterations"] for c in checks) == linear
             and sum(c["status"] != "converged" for c in checks) == summary["failed"])
    valid = valid and len({(c["step"], c["stage"], c["newton"]) for c in checks}) == len(checks)
    valid = valid and min(newton, last_newton, stages, linear, summary["failed"]) >= 0 and summary["dofs"] > 0
    for c in checks:
        finite = all(c[n] is not None for n in ("true_abs", "rhs_norm", "target"))
        valid = valid and c["status"] in ("converged", "nonfinite", "solver_failure", "iteration_limit", "residual_failed")
        valid = valid and min(c["step"], c["newton"], c["iterations"]) >= 0 and c["stage"] > 0
        valid = valid and c["at_limit"] in (0, 1)
        valid = valid and all(c[n] is None or c[n] >= 0 for n in ("true_abs", "rhs_norm", "target"))
        if c["status"] == "converged":
            valid = valid and finite and 0 <= c["true_abs"] < c["target"] and c["rhs_norm"] >= 0
        if c["status"] == "iteration_limit":
            valid = valid and c["at_limit"] == 1
    valid = valid and all(summary[k] is None or summary[k] >= 0 for k in
                         ("nonlinear_abs", "nonlinear_initial", "nonlinear_rtol", "nonlinear_atol"))
    if not valid:
        result["status"] = "invalid_evidence"
        return result
    finite = all(summary[k] is not None for k in
                 ("nonlinear_abs", "nonlinear_initial", "nonlinear_rtol", "nonlinear_atol"))
    nonlinear_ok = finite and summary["nonlinear_abs"] < max(
        summary["nonlinear_atol"], summary["nonlinear_rtol"] * summary["nonlinear_initial"])
    result["strict_nonlinear_convergence"] = nonlinear_ok
    if summary["failed"]:
        result["status"] = "linear_failure"
    elif not converged or not nonlinear_ok:
        result["status"] = "nonlinear_failure"
    elif returncode != 0:
        result["status"] = "driver_failure"
    elif "(SHORT OF TARGET)" in output or "vanka stencil vs bsr:" in output and "MISMATCH" in output:
        result["status"] = "verification_failure"
    else:
        result["status"] = "converged"
    return result


def configuration(cells, level, reynolds, pc, threads, max_iterations, restart, warp=0):
    if level not in (2, 4, 8, 16) or cells < 2 or cells % level:
        raise ValueError("fine cell counts must be positive multiples of a power-of-two SS level > 1")
    # The regularized cavity is [0,2]^3, so Re = rho U Ly / mu = 2 / mu.
    return {"SFEM_CASE": "cavity_reg", "SFEM_N": str(cells // level),
            "SFEM_NX": str(cells // level), "SFEM_NY": str(cells // level), "SFEM_NZ": str(cells // level),
            "SFEM_LX": "2", "SFEM_LY": "2", "SFEM_LZ": "2", "SFEM_RHO": "1", "SFEM_U": "1",
            "SFEM_MU": format(2.0 / reynolds, ".17g"), "SFEM_GEOM": "isoparam" if warp else "affine",
            "SFEM_CAVITY_WARP": format(warp, ".17g"),
            "SFEM_ELEMENT_REFINE_LEVEL": str(level), "SFEM_GMG": "0", "SFEM_PRECOND": pc,
            "SFEM_FGMRES": "1", "SFEM_FGMRES_RESTART": str(restart),
            "SFEM_LSOLVE_MAX_IT": str(max_iterations), "SFEM_LSOLVE_RTOL": "1e-6",
            "SFEM_LSOLVE_ATOL": "1e-14", "SFEM_LSOLVE_DTOL": "1e4",
            "SFEM_NL_MAX_IT": "30", "SFEM_NL_RTOL": "1e-8", "SFEM_NL_ATOL": "1e-12",
            "SFEM_NL_STOL": "0", "SFEM_NL_LS_FLOOR": "0", "SFEM_NL_LINESEARCH": "1",
            "SFEM_RE_ADAPT": "0", "SFEM_RE_STEP": "4", "SFEM_RE_MAX_RETRY": "0",
            "SFEM_MATRIX_FREE": "1", "SFEM_RC_EXACT_JAC": "1", "SFEM_CONV_HO": "0",
            "SFEM_PGRAD_CACHE": "1", "SFEM_PIN_PRESSURE": "0", "SFEM_PACK_SIZE": "1024",
            "SFEM_SS_PACK_SIZE": "0", "SFEM_PC_SCHUR": "0", "SFEM_GMG_OMEGA": "0.35",
            "SFEM_VANKA_PRECISION": "double", "SFEM_VANKA_MULT": "1", "SFEM_VANKA_FREEZE": "1",
            "SFEM_ENABLE_OUTPUT": "0", "SFEM_CHECK_LINEAR_RESIDUAL": "1",
            "OMP_NUM_THREADS": str(threads), "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "true",
            "OMP_PLACES": "cores", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def geometry_record(output, fraction):
    lines = [line for line in output.splitlines() if line.startswith("cavity_warp: ")]
    if len(lines) != 1:
        return None, False
    try:
        data = {key: float(value) for key, value in fields(lines[0]).items()}
        valid = (all(math.isfinite(value) for value in data.values())
                 and data["fraction"] == fraction and data["max_displacement"] > 0
                 and 0 <= data["jacobian_perturbation_bound"] < 1
                 and data["determinant_ratio_lower_bound"] > 0)
    except (ValueError, KeyError):
        return None, False
    valid = valid and bool(re.search(r"sscvfem: [1-9][0-9]* of [0-9]+ macro elements are curved", output))
    return data, valid


def capture(binary, directory, config, timeout):
    directory.mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SFEM_", "CVFEM_", "OMP_"))}
    env.update(config)
    command = [str(binary), str(directory / "unused-output")]
    log = directory / "run.log"
    timed_out = False
    with log.open("w") as stream:
        try:
            code = subprocess.call(command, env=env, cwd=str(directory), stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    output = log.read_text()
    try:
        result = parse_log(output, code)
    except (ValueError, KeyError) as error:
        result = {"status": "invalid_evidence", "parse_error": str(error), "returncode": code}
    if timed_out:
        result["status"] = "timeout"
    if result.get("driver_summary") is not None:
        summary = result["driver_summary"]
        configured = (summary["nonlinear_rtol"] == float(config["SFEM_NL_RTOL"])
                      and summary["nonlinear_atol"] == float(config["SFEM_NL_ATOL"]))
        for record in result["linear_checks"]:
            if record["rhs_norm"] is not None and record["target"] is not None:
                target = max(float(config["SFEM_LSOLVE_ATOL"]), float(config["SFEM_LSOLVE_RTOL"]) * record["rhs_norm"])
                configured = configured and math.isclose(record["target"], target, rel_tol=1e-14, abs_tol=0)
            configured = configured and record["iterations"] <= int(config["SFEM_LSOLVE_MAX_IT"])
            configured = configured and record["at_limit"] == int(record["iterations"] >= int(config["SFEM_LSOLVE_MAX_IT"]))
        if not configured:
            result["status"] = "invalid_evidence"
            result["parse_error"] = "reported tolerances or iteration limits differ from configuration"
    result["process_status"] = "timeout" if timed_out else "signal" if code < 0 else "exited"
    if float(config.get("SFEM_CAVITY_WARP", "0")) > 0:
        result["geometry"], geometry_ok = geometry_record(output, float(config["SFEM_CAVITY_WARP"]))
        if not geometry_ok:
            result["status"] = "geometry_failure"
    result.update(command=command, environment=config, log_sha256=digest(log), log=str(log.name))
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def provenance(repo, out, sources, supplied=None):
    revision = None
    patch = ""
    if (repo / ".git").exists():
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=str(repo), universal_newlines=True).strip()
        if Path(root).resolve() != repo.resolve():
            raise ValueError("source directory is not the Git root")
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), universal_newlines=True).strip()
        patch = subprocess.check_output(["git", "diff", "HEAD", "--", "spikes/cvfem/src/drivers/cvfem_hex8_ns_ssgmg.cpp"],
                                        cwd=str(repo), universal_newlines=True)
    elif supplied is None:
        raise ValueError("source-only bundles require --provenance-file (for example SOURCE-REVISIONS)")
    data = {"source_mode": "git" if revision else "source_only", "git_revision": revision}
    if supplied is not None:
        destination = out / "SOURCE-REVISIONS"
        destination.write_bytes(supplied.read_bytes())
        data["supplied_revisions_sha256"] = digest(destination)
    (out / "solver-driver.patch").write_text(patch)
    data["driver_patch_sha256"] = digest(out / "solver-driver.patch")
    data["sources"] = {str(path.relative_to(repo)): digest(path) for path in sources}
    for path in sources:
        destination = out / "sources" / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    return data


def self_test():
    good = ("linear_check: step=0 stage=1 newton=0 iterations=3 status=converged "
            "true_abs=1e-9 rhs_norm=1 target=1e-6 at_limit=0\n"
            "newton_converged: 1  newton_it: 1 (last stage)  newton_total: 1 over 1 stage(s)  lin_it_total: 3\n"
            "linear_check_summary: checked=1 failed=0 nonlinear_abs=1e-10 nonlinear_initial=1 "
            "nonlinear_rtol=1e-8 nonlinear_atol=1e-12 dofs=500\n")
    assert parse_log(good, 0)["status"] == "converged"
    assert parse_log(good, 1)["status"] == "driver_failure"
    assert parse_log(good.replace("true_abs=1e-9", "true_abs=1e-3"), 0)["status"] == "invalid_evidence"
    assert parse_log(good.replace("true_abs=1e-9", "true_abs=nan"), 0)["status"] == "invalid_evidence"
    assert parse_log(good.replace("lin_it_total: 3", "lin_it_total: 4"), 0)["status"] == "invalid_evidence"
    assert parse_log(good.replace("nonlinear_abs=1e-10", "nonlinear_abs=1e-5"), 0)["status"] == "nonlinear_failure"
    assert parse_log(good.replace("nonlinear_abs=1e-10", "nonlinear_abs=-1e-10"), 0)["status"] == "invalid_evidence"
    duplicate = good.splitlines()[0] + "\n" + good
    duplicate = duplicate.replace("checked=1", "checked=2").replace("lin_it_total: 3", "lin_it_total: 6")
    assert parse_log(duplicate, 0)["status"] == "invalid_evidence"
    capped = good.replace("status=converged", "status=iteration_limit").replace("true_abs=1e-9", "true_abs=1e-2")
    capped = capped.replace("failed=0", "failed=1").replace("at_limit=0", "at_limit=1")
    assert parse_log(capped, 1)["status"] == "linear_failure"
    assert parse_log(good + "continuation: (SHORT OF TARGET)\n", 0)["status"] == "verification_failure"
    assert parse_log("", 0)["status"] == "incomplete"
    for level in (0, -1, 3):
        try:
            configuration(4, level, 1, "bjacobi", 1, 1000, 60)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid SS level was accepted")
    for output in ("", "cavity_warp: fraction=0.025\n", "cavity_warp: fraction=bad\n"):
        assert not geometry_record(output, 0.025)[1]
    from unittest.mock import patch
    with tempfile.TemporaryDirectory(prefix="solver-geometry-") as directory:
        config = configuration(4, 2, 1, "bjacobi", 1, 1000, 60, .025)
        for index, geometry in enumerate(("", "cavity_warp: fraction=bad\n")):
            def output_command(*args, **kwargs):
                kwargs["stdout"].write(good + geometry)
                return 0
            with patch.object(subprocess, "call", output_command):
                result = capture(Path("fixture"), Path(directory) / str(index), config, 1)
            assert result["status"] == "geometry_failure"
            assert result["returncode"] == 0 and result["process_status"] == "exited"
    with tempfile.TemporaryDirectory(prefix="solver-provenance-") as directory:
        root = Path(directory)
        source, out = root / "source", root / "out"
        source.mkdir()
        out.mkdir()
        revisions = root / "SOURCE-REVISIONS"
        revisions.write_text("test source bundle revision\n")
        sample = source / "public.cpp"
        sample.write_text("int main() { return 0; }\n")
        record = provenance(source, out, [sample], revisions)
        assert record["source_mode"] == "source_only" and record["git_revision"] is None
        assert (out / "sources/public.cpp").read_bytes() == sample.read_bytes()
        try:
            provenance(source, out, [sample])
        except ValueError:
            pass
        else:
            raise AssertionError("missing source-only provenance was accepted")
    print("solver table parser: convergence, residual, cap, nonfinite and missing evidence checks PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sizes", default="4,8,12", help="fine cells in each direction")
    parser.add_argument("--reynolds", default="1,10")
    parser.add_argument("--warps", default="0,0.025", help="boundary-preserving displacement / box size")
    parser.add_argument("--provenance-file", type=Path, help="bundle SOURCE-REVISIONS; required without .git")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--restart", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.binary is None or args.out is None:
        parser.error("--binary and --out are required")
    binary, out = args.binary.resolve(), args.out.resolve()
    if not binary.is_file():
        parser.error("binary does not exist")
    if any(value < 1 for value in (args.threads, args.max_iterations, args.restart, args.timeout)):
        parser.error("threads, iteration limit, restart and timeout must be positive")
    sizes = list(map(int, args.sizes.split(",")))
    reynolds = list(map(float, args.reynolds.split(",")))
    warps = list(map(float, args.warps.split(",")))
    if any(not math.isfinite(value) or value <= 0 for value in reynolds):
        parser.error("Reynolds numbers must be finite and positive")
    if any(not math.isfinite(value) or value < 0 or value > 0.05 for value in warps):
        parser.error("warp fractions must be finite and between 0 and 0.05")
    if len(sizes) != len(set(sizes)) or len(reynolds) != len(set(reynolds)) or len(warps) != len(set(warps)):
        parser.error("sizes, Reynolds numbers and warp fractions must not repeat")
    for cells in sizes:
        configuration(cells, args.level, reynolds[0], "bjacobi", args.threads, args.max_iterations, args.restart)
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    sources = [Path(__file__).resolve(), repo / "spikes/cvfem/src/drivers/cvfem_hex8_ns_ssgmg.cpp",
               repo / "spikes/cvfem/src/support/cvfem_fgmres.hpp",
               repo / "spikes/cvfem/src/frontend/galerkin/cvfem_ss_vanka.cpp"]
    manifest = {"binary_sha256": digest(binary), "host": platform.node(), "platform": platform.platform(),
                "collected_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "scope": "CPU first-order exact-RC matrix-free regularized cavity; same semi-structured mesh per pair; no timings accepted",
                "preconditioners": {"bjacobi": "3x3 velocity block and scalar pressure inverse per node; omega=1",
                                    "vanka": "multiplicative diagonal Vanka; 32-dof microcell patches; FP64 storage; omega=0.35"}}
    manifest.update(provenance(repo, out, sources, args.provenance_file))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    cases = [(cells, reynolds_number, warp) for cells in sizes for reynolds_number in reynolds for warp in warps]
    for i, (cells, reynolds_number, warp) in enumerate(cases):
        order = ("bjacobi", "vanka") if i % 2 == 0 else ("vanka", "bjacobi")
        for pc in order:
            name = "n{}-re{:g}-warp{:g}-{}".format(cells, reynolds_number, warp, pc)
            config = configuration(cells, args.level, reynolds_number, pc, args.threads,
                                   args.max_iterations, args.restart, warp)
            result = capture(binary, out / name, config, args.timeout)
            row = {"fine_cells": cells, "macro_cells": cells // args.level, "level": args.level,
                   "reynolds": reynolds_number, "warp": warp, "preconditioner": pc, "directory": name, **result}
            rows.append(row)
            (out / "table.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
            print("{}: {} Newton={} linear={}".format(name, result["status"],
                  result.get("newton_iterations", "?"), result.get("linear_iterations", "?")), flush=True)
    lines = ["CPU regularized-cavity iteration study. Failed and capped runs are retained.", "",
             "| Fine cells | Re | Warp | Preconditioner | Newton | Linear | Status |",
             "|---:|---:|---:|---|---:|---:|---|"]
    for row in rows:
        lines.append("| {} | {:g} | {:g} | {} | {} | {} | {} |".format(row["fine_cells"], row["reynolds"], row["warp"],
                     row["preconditioner"], row.get("newton_iterations", "?"),
                     row.get("linear_iterations", "?"), row["status"]))
    lines.extend(["", "Counts include every attempted continuation stage. Failure counts are work spent, not converged iteration counts.",
                  "This measures the named semi-structured solver configuration, not unstructured packed solver performance or mesh-independent convergence."])
    (out / "table.md").write_text("\n".join(lines) + "\n")
    return 0 if all(row["status"] == "converged" for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
