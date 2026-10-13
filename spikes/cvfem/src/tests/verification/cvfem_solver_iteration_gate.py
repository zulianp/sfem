#!/usr/bin/env python3
"""Check the opt-in linear residual gate on a small public cavity."""

import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/solver_iteration_study.py"
spec = importlib.util.spec_from_file_location("solver_iteration_study", str(SCRIPT))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    out = args.out.resolve() if args.out else Path(tempfile.mkdtemp(prefix="cvfem-solver-gate-"))
    out.mkdir(parents=True, exist_ok=True)
    study.self_test()
    config = study.configuration(4, 2, 1, "bjacobi", 1, 1000, 60)
    checked = study.capture(binary, out / "checked", config, 120)
    assert checked["status"] == "converged", checked
    normal = dict(config, SFEM_CHECK_LINEAR_RESIDUAL="0")
    baseline = study.capture(binary, out / "baseline", normal, 120)
    assert baseline["returncode"] == 0, baseline
    match = study.SUMMARY.search((out / "baseline/run.log").read_text())
    assert match and int(match.group(3)) == checked["newton_iterations"]
    assert int(match.group(5)) == checked["linear_iterations"]
    capped = study.capture(binary, out / "capped", dict(config, SFEM_LSOLVE_MAX_IT="1"), 120)
    assert capped["returncode"] != 0 and capped["status"] == "linear_failure", capped
    assert capped["linear_checks"][0]["status"] == "iteration_limit", capped
    invalid = study.capture(binary, out / "invalid", dict(config, SFEM_LSOLVE_RTOL="nan"), 120)
    assert invalid["returncode"] != 0 and not invalid.get("linear_checks"), invalid
    vanka = study.capture(binary, out / "vanka", dict(config, SFEM_PRECOND="vanka", SFEM_VANKA_CHECK="1"), 120)
    assert vanka["status"] == "converged", vanka
    vanka_log = (out / "vanka/run.log").read_text()
    assert "vanka stencil vs bsr:" in vanka_log and "MISMATCH" not in vanka_log
    for pc in ("bjacobi", "vanka"):
        warped_config = study.configuration(4, 2, 1, pc, 1, 1000, 60, warp=0.025)
        warped = study.capture(binary, out / ("warped-" + pc), warped_config, 120)
        assert warped["status"] == "converged", warped
        assert warped["geometry"]["determinant_ratio_lower_bound"] > 0
    print("CPU solver residual gate: unchanged counts, both preconditioners, capped and nonfinite controls PASS")
    print("Logs retained at {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
