#!/usr/bin/env python3
"""Characterise the Prony-series material by sweeping the severity of what it approximates.

A single run tells you the solver converged. It does not tell you how far the answer is from
the model the case asked for, or where that model stops describing the material. Three severity
axes do, each with a control at zero severity:

  --sweep dt      Numerical severity: dt/tau_fast, how coarsely the fastest relaxation mode is
                  resolved by the recursion alpha_i = exp(-dt/tau_i). The recovered weights must
                  converge as dt -> 0, and the rate they converge at is the measurement of the
                  recursion's time-integration error. Zero severity is dt -> 0.

  --sweep angle   Physical severity: how far the twist is from the linear regime the Prony
                  superposition assumes. The material is quasi-linear, so the recovered weights
                  should be nearly angle-independent at small twist and depart at large twist.
                  Where they depart is the useful number, and it is not otherwise known.

  --sweep bulk    Model severity: the Prony series multiplies only the DEVIATORIC stress, so the
                  torque is g(t) * T_dev + T_vol with an unrelaxed volumetric part that the fit
                  cannot help absorbing into g_inf. Sweeping K measures how much that costs. It
                  is the reason the recovered weights are close to, but never exactly, the ones
                  the case asked for. Zero severity is K -> 0.

  --sweep frequency  The lag itself, swept: a simulated DMA. Drives the twist sinusoidally at a
                  range of periods and reads the loss angle off each, against the closed-form
                  Prony prediction tan(delta) = G''(w)/G'(w). Zero severity is either end of the
                  range -- at w*tau << 1 the material relaxes fully within a cycle and at
                  w*tau >> 1 it has no time to relax at all, and the lag vanishes in both limits.
                  The peak in between is the signature of the relaxation times themselves.

  --sweep control Zero severity in both senses at once: no Prony terms at all. An elastic solid
                  does not relax, so the torque through the hold must be constant. If this run
                  shows any decay the instrument is broken and nothing above it means anything.

Every row reports the problem size and the thread count alongside the timing, because a
duration without them is not a result.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_torsion_release import fit_relaxation, read_history  # noqa: E402
from analyse_hysteresis import analyse as analyse_hysteresis, prony_complex_modulus  # noqa: E402

import yaml


def run_case(exe: Path, workdir: Path, base: dict, name: str, overrides) -> dict:
    """Writes a case derived from `base`, runs it, and returns the measurement."""
    case = copy.deepcopy(base)
    overrides(case)
    case["output"]["path"] = f"results_{name}"
    case["verbose"] = False

    case_path = workdir / f"{name}.yaml"
    case_path.write_text(yaml.safe_dump(case, sort_keys=False))

    log = workdir / f"{name}.log"
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")

    start = time.time()
    with log.open("w") as fh:
        rc = subprocess.run([str(exe), case_path.name], cwd=workdir, stdout=fh, stderr=subprocess.STDOUT).returncode
    wall = time.time() - start

    text = log.read_text()
    size = re.search(r"^nnodes: (\d+)\s+nelements: (\d+)\s+ndof: (\d+)", text, re.M)
    total_lin = re.search(r"^Total linear iterations: (\d+)", text, re.M)

    row = {
        "name": name,
        "ok": rc == 0,
        "wall": wall,
        "ndof": int(size.group(3)) if size else 0,
        "nelements": int(size.group(2)) if size else 0,
        "linear_its": int(total_lin.group(1)) if total_lin else 0,
        "threads": int(env["OMP_NUM_THREADS"]),
    }

    if rc != 0:
        return row

    rows = read_history(str(workdir / case["output"]["path"] / case["output"].get("history_csv", "history.csv")))
    held = [r for r in rows if not r.released]
    if not held:
        return row

    final_angle = held[-1].angle
    plateau = [r for r in held if abs(r.angle - final_angle) <= 1e-12 * max(1.0, abs(final_angle))]

    peak = max(abs(r.torque) for r in held)
    row["peak_torque"] = peak
    row["relaxed_fraction"] = (abs(plateau[0].torque) - abs(plateau[-1].torque)) / peak if plateau else 0.0

    terms = (case.get("material") or {}).get("prony") or []
    if terms and len(plateau) >= 2 * (len(terms) + 1):
        taus = [float(t["tau"]) for t in terms]
        ramp = float((case.get("torsion") or {}).get("ramp_time", 0.0))
        weights, rms = fit_relaxation(plateau, taus, ramp)
        row["weights"] = weights
        row["fit_rms"] = rms
        row["asked"] = [1.0 - sum(float(t["g"]) for t in terms)] + [float(t["g"]) for t in terms]

    return row


def print_table(title: str, rows, axis_label: str, axis_values) -> None:
    print()
    print(title)
    print("=" * len(title))
    if rows and rows[0]["ndof"]:
        print(
            f"{rows[0]['ndof']} dof, {rows[0]['nelements']} HEX8 elements, "
            f"{rows[0]['threads']} thread(s)"
        )

    header = f"{axis_label:>10}  {'wall/s':>8}  {'lin its':>10}  {'relaxed':>8}  {'fit rms':>10}  weights (g_inf, g_i...)"
    print(header)
    print("-" * len(header))

    for value, row in zip(axis_values, rows):
        if not row["ok"]:
            print(f"{value:>10}  FAILED -- see {row['name']}.log")
            continue
        weights = row.get("weights")
        wtxt = "  ".join(f"{w:.4f}" for w in weights) if weights else "-"
        print(
            f"{value:>10}  {row['wall']:>8.1f}  {row['linear_its']:>10d}  "
            f"{row.get('relaxed_fraction', 0):>7.2%}  {row.get('fit_rms', float('nan')):>10.2e}  {wtxt}"
        )

    if rows and rows[0].get("asked"):
        print(f"{'asked':>10}  {'':>8}  {'':>10}  {'':>8}  {'':>10}  " + "  ".join(f"{w:.4f}" for w in rows[0]["asked"]))


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["dt", "angle", "bulk", "frequency", "control", "all"], default="all")
    ap.add_argument("--exe", default=str(here / "build" / "prony_visco_torsion"))
    ap.add_argument("--case", default=str(here / "cases" / "torsion_release.yaml"))
    ap.add_argument("--mesh", help="an existing SFEM HEX8 mesh folder; one is built if omitted")
    ap.add_argument("--workdir", default=str(here / "sweep"))
    args = ap.parse_args()

    exe = Path(args.exe)
    if not exe.is_file():
        print(f"missing executable: {exe}", file=sys.stderr)
        return 1

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    if args.mesh:
        shutil.copytree(args.mesh, workdir / "mesh")
    else:
        python = os.environ.get("PRONY_PYTHON", str(here.parent.parent / "venv" / "bin" / "python"))
        subprocess.run(
            [
                python,
                str(here.parent.parent / "python" / "sfem" / "mesh" / "box_mesh.py"),
                str(workdir / "mesh"),
                "--cell_type=HEX8",
                "-x", "12", "-y", "5", "-z", "5",
                "--width", "1.0", "--height", "0.2", "--depth", "0.2",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    base = yaml.safe_load(Path(args.case).read_text())
    base["mesh"] = "mesh"
    for cond in base.get("dirichlet_conditions", []):
        cond["path"] = cond["path"].replace("mesh/", "mesh/", 1)
    base["torsion"]["sideset"] = "mesh/surface/sidesets/right"

    # The release is not part of this measurement -- the weights are recovered from the hold --
    # and dropping it keeps each sweep point cheap.
    base["torsion"]["release"] = {"mode": "hold"}
    base["output"]["export_freq"] = 1000

    rc = 0

    if args.sweep in ("dt", "all"):
        values = [0.4, 0.2, 0.1, 0.05]
        rows = []
        for dt in values:
            rows.append(
                run_case(exe, workdir, base, f"dt_{dt}", lambda c, dt=dt: c["time"].__setitem__("dt", dt))
            )
        print_table(
            "Numerical severity: dt at fixed tau = {1, 10}, so dt/tau_fast spans a decade",
            rows,
            "dt",
            values,
        )

    if args.sweep in ("angle", "all"):
        values = [0.15, 0.3, 0.6, 1.2]
        rows = []
        for angle in values:
            rows.append(
                run_case(
                    exe, workdir, base, f"angle_{angle}", lambda c, a=angle: c["torsion"].__setitem__("angle", a)
                )
            )
        print_table(
            "Physical severity: twist angle, testing how far the quasi-linear ansatz holds",
            rows,
            "angle/rad",
            values,
        )

    if args.sweep in ("bulk", "all"):
        values = [5.0, 50.0, 500.0]
        rows = []
        for k in values:
            rows.append(run_case(exe, workdir, base, f"K_{k:g}", lambda c, k=k: c["material"].__setitem__("K", k)))
        print_table(
            "Model severity: bulk modulus, which the Prony series does not relax",
            rows,
            "K",
            values,
        )

    if args.sweep in ("frequency", "all"):
        periods = [60.0, 20.0, 12.0, 4.0, 2.0]
        amplitude = 0.15
        terms = (base.get("material") or {}).get("prony") or []
        g = [float(t["g"]) for t in terms]
        tau = [float(t["tau"]) for t in terms]

        print()
        title = "The lag itself: a frequency sweep, i.e. a simulated DMA"
        print(title)
        print("=" * len(title))
        header = (f"{'period/s':>9} {'omega':>8} {'wall/s':>8} {'delta/deg':>10} {'predicted':>10} "
                  f"{'ratio':>7} {'tan d':>8} {'loop area':>12} {'fit rms':>9}")
        printed_size = False

        for period in periods:
            def cyclic(c, period=period, amplitude=amplitude):
                c["torsion"]["profile"] = "cyclic"
                c["torsion"]["period"] = period
                c["torsion"]["angle"] = amplitude
                c["torsion"].pop("ramp_time", None)
                # 80 samples per cycle, three cycles: the first two let the transient from the
                # impulsive start decay, the last one is what gets fitted.
                c["time"] = {"dt": period / 80.0, "t_end": 3 * period}

            row = run_case(exe, workdir, base, f"f_{period:g}", cyclic)
            if not printed_size:
                print(f"{row['ndof']} dof, {row['nelements']} HEX8 elements, {row['threads']} thread(s)")
                print(header)
                print("-" * len(header))
                printed_size = True

            if not row["ok"]:
                print(f"{period:>9.1f}  FAILED -- see {row['name']}.log")
                rc = 1
                continue

            hist = workdir / f"results_f_{period:g}" / "history.csv"
            rows = [(r.time, r.angle, r.torque) for r in read_history(str(hist))]
            m = analyse_hysteresis(rows, period, amplitude)
            storage, loss = prony_complex_modulus(g, tau, m["omega"])
            predicted = math.degrees(math.atan2(loss, storage))
            print(f"{period:>9.1f} {m['omega']:>8.4f} {row['wall']:>8.1f} {m['delta_deg']:>10.3f} "
                  f"{predicted:>10.3f} {m['delta_deg']/predicted:>7.4f} {m['tan_delta']:>8.4f} "
                  f"{m['loop_area']:>12.4e} {m['rms']:>9.2e}")

    if args.sweep in ("control", "all"):
        def no_prony(c):
            c["material"].pop("prony", None)

        row = run_case(exe, workdir, base, "control_elastic", no_prony)
        print()
        print("Zero-severity control: no Prony terms")
        print("=====================================")
        if not row["ok"]:
            print(f"FAILED -- see {row['name']}.log")
            rc = 1
        else:
            relaxed = row.get("relaxed_fraction", 0.0)
            print(
                f"{row['ndof']} dof, {row['nelements']} HEX8 elements, {row['threads']} thread(s), "
                f"{row['wall']:.1f} s, {row['linear_its']} linear iterations"
            )
            print(f"torque relaxed while held: {relaxed:.3%} (an elastic solid must not relax at all)")
            if abs(relaxed) > 1e-6:
                print("CONTROL FAILED: an elastic solid relaxed", file=sys.stderr)
                rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
