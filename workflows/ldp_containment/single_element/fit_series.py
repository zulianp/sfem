#!/usr/bin/env python3
"""Fit the valves' material to 70 EPDM 281: the relaxed shear modulus and the slow terms' weights.

    fit_series.py <playground> <measurements> <out> [--slow 4] [--steps 1000] [--dma-temp 30]
                  [--dma-weight 1]

The relaxation times stay those of the series in simulation 2's case file; so do the absolute weights
mu_0 g_i of its fast terms (all but the --slow slowest). Free are mu_inf = mu_0 g_inf and the slow
terms' absolute weights a_i = mu_0 g_i, fitted in log space by least squares (scipy) to
  - the uniaxial, equibiaxial and planar tests (measured_tests.py: one HEX8 element following each
    measured strain history, quasi-static), the stress at every measured instant over the test's
    largest measured stress, each test weighted 1/sqrt(its samples);
  - the storage and loss moduli at --dma-temp from 0.1 to 10 Hz, relative, small strain (the
    complex Young's modulus of mu(omega) and K), weighted --dma-weight/sqrt(their count).
K stays tied to mu_inf by nu = 0.4, C01/C10 stays 4. <measurements> holds uniaxial.csv, equibiax.csv,
pureshear.csv and dma.csv (epdm281_extract.py). <out> must not exist; writes <out>/fit.yaml (the
material block, the parameters, and the misfit before and after) and <out>/evals.csv.
"""

import argparse
import concurrent.futures
import csv
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import least_squares
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measured_tests import TESTS, run  # noqa: E402
from stress_strain import ROOT, SIM2  # noqa: E402

NU = 0.4


def base_series():
    """Simulation 2's (tau_i, mu_0 g_i) slowest first, and its mu_inf, in Pa."""
    m = yaml.safe_load(SIM2.read_text())["material"]
    series = sorted(((float(p["tau"]), float(p["g"])) for p in m["relaxation"]["iso"]["prony"]), reverse=True)
    mu0 = 2 * (float(m["C10"]) + float(m["C01"]))
    return [(tau, mu0 * g) for tau, g in series], mu0 * (1 - sum(g for _, g in series))


def material(mu_inf, weights):
    """The case's material block from mu_inf and the absolute weights [(tau, a)] (Pa)."""
    mu0 = mu_inf + sum(a for _, a in weights)
    k = mu_inf * 2 * (1 + NU) / (3 * (1 - 2 * NU))
    return {"type": "GeneratedMooneyRivlinProny", "C10": mu0 / 10, "C01": 4 * mu0 / 10, "K": k,
            "relaxation": {"iso": {"prony": [{"g": a / mu0, "tau": tau} for tau, a in weights],
                                   "history_storage": "full"}}}


def young(mu_inf, weights, frequency):
    """The small-strain complex Young's modulus (Pa): the shear modulus relaxes, K does not."""
    w = 2 * math.pi * frequency
    mu = mu_inf + sum(a * 1j * w * tau / (1 + 1j * w * tau) for tau, a in weights)
    k = mu_inf * 2 * (1 + NU) / (3 * (1 - 2 * NU))
    return 9 * k * mu / (3 * k + mu)


def interp(run_rows, t):
    pts = [(0.0, 0.0)] + run_rows
    for (t0, s0), (t1, s1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            return s0 + (s1 - s0) * (t - t0) / (t1 - t0)
    return pts[-1][1]


def dissipated(pts):
    """The share of the largest work a loop over (strain, stress) dissipates."""
    work = [0.0]
    for (e0, s0), (e1, s1) in zip(pts, pts[1:]):
        work.append(work[-1] + (e1 - e0) * (s0 + s1) / 2)
    return work[-1] / max(work)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("playground", type=Path)
    p.add_argument("measurements", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--slow", type=int, default=4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dma-temp", type=float, default=30)
    p.add_argument("--dma-weight", type=float, default=1.0)
    a = p.parse_args()
    if a.out.exists():
        sys.exit("fit_series: %s exists; give a new folder" % a.out)
    a.out.mkdir(parents=True)
    mesh = a.out / "mesh"
    subprocess.run([sys.executable, str(ROOT / "python/sfem/mesh/box_mesh.py"), str(mesh), "--cell_type=HEX8",
                    "-x", "2", "-y", "2", "-z", "2", "--width", "1", "--height", "1", "--depth", "1"], check=True)
    measured, histories = {}, {}
    for test in TESTS:
        rows = [(float(r["t"]), float(r["strain"]), float(r["stress"]) * 1e6)
                for r in csv.DictReader(open(a.measurements / (test + ".csv")))]
        measured[test] = rows
        hist = [(t, e) for t, e, _ in rows]
        histories[test] = [r for i, r in enumerate(hist) if i == 0 or r[0] > hist[i - 1][0]]
    dma = [r for r in ({k: float(v) for k, v in r.items()} for r in csv.DictReader(open(a.measurements / "dma.csv")))
           if r["temperature"] == a.dma_temp]
    series, mu_inf0 = base_series()
    slow, fast = series[:a.slow], series[a.slow:]
    x0 = np.log([mu_inf0] + [w for _, w in slow])
    evals = a.out / "evals"
    evals.mkdir()
    log = []

    def unpack(x):
        v = [float(z) for z in np.exp(x)]
        return v[0], [(tau, w) for (tau, _), w in zip(slow, v[1:])] + fast

    def simulate(x):
        mu_inf, weights = unpack(x)
        mat = material(mu_inf, weights)
        root = evals / ("%04d" % len(log))
        root.mkdir()
        with concurrent.futures.ThreadPoolExecutor(len(TESTS)) as pool:
            done = list(pool.map(lambda t: run(a.playground.resolve(), root, mesh.resolve(), t, 0, histories[t],
                                               a.steps, mat), TESTS))
        return mu_inf, weights, {test: rows for test, _, rc, complete, rows in done if rc == 0 and complete}

    def residual(x, record=True):
        mu_inf, weights, runs = simulate(x)
        res, report = [], {}
        for test in TESTS:
            rows = measured[test]
            top = max(sv for _, _, sv in rows)
            if test not in runs:
                res += [10.0] * (len(rows) - 1)
                report[test] = None
                continue
            r = [(interp(runs[test], t) - sv) / top for t, _, sv in rows if t > 0]
            res += [v / math.sqrt(len(r)) for v in r]
            model = [(0.0, 0.0)] + [(e, interp(runs[test], t)) for t, e, _ in rows if t > 0]
            report[test] = {"rms": math.sqrt(sum(v * v for v in r) / len(r)),
                            "dissipated_model": dissipated(model),
                            "dissipated_measured": dissipated([(e, sv) for _, e, sv in rows])}
        for r in dma:
            e = young(mu_inf, weights, r["frequency"]) / 1e6
            res += [a.dma_weight * (e.real / r["storage"] - 1) / math.sqrt(2 * len(dma)),
                    a.dma_weight * (e.imag / r["loss"] - 1) / math.sqrt(2 * len(dma))]
            report["dma_%g" % r["frequency"]] = {"storage": e.real, "loss": e.imag,
                                                 "storage_measured": r["storage"], "loss_measured": r["loss"]}
        cost = float(sum(v * v for v in res))
        if record:
            log.append([cost, mu_inf] + [w for _, w in weights[:a.slow]])
        return np.array(res), report, mu_inf, weights

    def fun(x):
        return residual(x)[0]

    _, before, _, _ = residual(x0)
    fit = least_squares(fun, x0, diff_step=1e-3, x_scale=1.0, max_nfev=60)
    res, after, mu_inf, weights = residual(fit.x)
    out = {"material": material(mu_inf, weights),
           "mu_inf": mu_inf, "slow_weights": [{"tau": tau, "a": w} for tau, w in weights[:a.slow]],
           "cost_before": float(log[0][0]), "status": int(fit.status), "message": str(fit.message), "nfev": int(fit.nfev),
           "cost": float(sum(v * v for v in res)), "before": before, "after": after}
    (a.out / "fit.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    with open(a.out / "evals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cost", "mu_inf"] + ["a_%d" % i for i in range(1, a.slow + 1)])
        w.writerows(log)
    print("mu_inf %.4g Pa (was %.4g); slow weights %s" % (mu_inf, mu_inf0, ", ".join(
        "%.3g (was %.3g) at %g s" % (w, w0, tau) for (tau, w), (_, w0) in zip(weights[:a.slow], slow))))
    for test in TESTS:
        b, f_ = before[test], after[test]
        print("%-10s rms %.3f -> %s  dissipated %.2f -> %s (measured %.2f)" % (
            test, b["rms"], "%.3f" % f_["rms"] if f_ else "failed", b["dissipated_model"],
            "%.2f" % f_["dissipated_model"] if f_ else "-", b["dissipated_measured"]))
    for key in sorted(k for k in after if k.startswith("dma")):
        r = after[key]
        print("%s Hz: E' %.2f (measured %.2f), E'' %.3f (measured %.3f)" % (
            key[4:], r["storage"], r["storage_measured"], r["loss"], r["loss_measured"]))


if __name__ == "__main__":
    main()
