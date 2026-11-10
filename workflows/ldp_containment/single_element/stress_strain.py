#!/usr/bin/env python3
"""The valves' EPDM on one element: engineering stress against engineering strain.

One HEX8 unit cube with three symmetry planes (x, y, z = 0 held normal), its face x = 1 pulled to
strain EPS and back at a constant strain rate, quasi-static, the history in double. Per rate, the
full 15-term series of simulation 2 and the series truncated to its K slowest terms, rescaled to the
same relaxation beyond them (g_i/D, C10 and C01 times D, D = g_inf + g_1 + ... + g_K; the four-term
series of the paper's step study is K = 4). A run whose Newton fails is recorded as such.

    stress_strain.py <playground> <out> [--rates 1e1,1e5,1e7,1e9] [--terms 4,5,6,15] [--eps 0.5]
                     [--steps 200]

<out> must not exist. Writes <out>/curves.csv (terms, rate, step, t, strain, stress), and
<out>/runs.csv (terms, rate, dt, steps, stored, rc, complete).
"""

import argparse
import concurrent.futures
import csv
import os
from pathlib import Path
import subprocess
import sys

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM2 = ROOT / "workflows/ldp_containment/transient/FF200_ExodusMesh/sim2.yaml"


def material(terms):
    """Simulation 2's material with its series truncated to the `terms` slowest, rescaled."""
    m = yaml.safe_load(SIM2.read_text())["material"]
    # PyYAML reads 1.460e8 as a string (YAML 1.1 wants a sign in the exponent).
    series = sorted(({"g": float(p["g"]), "tau": float(p["tau"])} for p in m["relaxation"]["iso"]["prony"]),
                    key=lambda p: -p["tau"])
    m.update({k: float(m[k]) for k in ("C10", "C01", "K")})
    g_inf = 1 - sum(p["g"] for p in series)
    kept = series[:terms]
    d = g_inf + sum(p["g"] for p in kept)
    return {"type": m["type"], "C10": m["C10"] * d, "C01": m["C01"] * d, "K": m["K"],
            "relaxation": {"iso": {"prony": [{"g": p["g"] / d, "tau": p["tau"]} for p in kept],
                                   "history_storage": "full"}}}


def case(mesh, terms, rate, eps, steps):
    ramp = eps / rate
    return {
        "mesh": {"path": str(mesh), "packed": False,
                 "sidesets": [{"name": n, "plane": {"axis": a, "at": w}}
                              for n, a, w in (("pull", "x", "max"), ("sym_x", "x", "min"),
                                              ("sym_y", "y", "min"), ("sym_z", "z", "min"))]},
        "material": material(terms),
        # Condition 0 is the pulled face: its reaction is the force.
        "dirichlet_conditions": [
            {"type": "sideset", "format": "name", "name": "pull", "component": 0, "value": eps,
             "profile": {"type": "tabulated", "path": "profile.csv"}},
            {"type": "sideset", "format": "name", "name": "sym_x", "component": 0, "value": 0},
            {"type": "sideset", "format": "name", "name": "sym_y", "component": 1, "value": 0},
            {"type": "sideset", "format": "name", "name": "sym_z", "component": 2, "value": 0}],
        "regime": "quasi_static",
        "time": {"dt": ramp / steps, "steps": 2 * steps, "max_cutbacks": 8},
        "solver": {"newton": {"max_it": 40, "atol": 1.0e-6, "rtol": 1.0e-12, "verbose": False,
                              "tangent_projection": "on_failure"},
                   "linear": {"type": "cg", "preconditioner": "jacobi", "rtol": 1.0e-12, "max_it": 1000}},
        "output": {"path": "out", "export_freq": 2 * steps, "quantities": True,
                   "history_csv": "history.csv", "probes": [{"type": "prony"}]},
    }


def run(playground, root, mesh, terms, rate, eps, steps):
    w = root / ("k%02d_r%.0e" % (terms, rate))
    w.mkdir()
    ramp = eps / rate
    (w / "profile.csv").write_text("0, 0\n%.17g, 1\n%.17g, 0\n" % (ramp, 2 * ramp))
    (w / "case.yaml").write_text(yaml.safe_dump(case(mesh, terms, rate, eps, steps), sort_keys=False))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with open(w / "run.log", "w") as log:
        rc = subprocess.run([str(playground), "case.yaml", "--set", "execution_space=host"], cwd=w, env=env,
                            stdout=log, stderr=subprocess.STDOUT).returncode
    rows = []
    q = w / "out" / "quantities.yaml"
    if q.exists():
        for e in (yaml.safe_load(q.read_text()) or {}).get("material_objective_history") or []:
            force = [c["resultant"] for c in e["constraint_reactions"] if c["condition"] == 0][0]
            rows.append((float(e["time"]), float(force)))
    stored = ""
    h = w / "history.csv"
    if h.exists():
        last = list(csv.DictReader(open(h)))
        if last:
            stored = next((v for k, v in last[-1].items() if k.endswith("stored")), "")
    complete = len(rows) >= 2 * steps
    return terms, rate, ramp / steps, steps, stored, rc, complete, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("playground", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--rates", default="1e1,1e5,1e7,1e9")
    p.add_argument("--terms", default="4,5,6,15")
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=200, help="steps per ramp, up and down")
    p.add_argument("--jobs", type=int, default=16)
    a = p.parse_args()
    if a.out.exists():
        sys.exit("stress_strain: %s exists; give a new folder" % a.out)
    a.out.mkdir(parents=True)
    mesh = a.out / "mesh"
    # box_mesh.py counts nodes per direction: two make one element.
    subprocess.run([sys.executable, str(ROOT / "python/sfem/mesh/box_mesh.py"), str(mesh), "--cell_type=HEX8",
                    "-x", "2", "-y", "2", "-z", "2", "--width", "1", "--height", "1", "--depth", "1"], check=True)
    runs = a.out / "runs"
    runs.mkdir()
    todo = [(k, r) for k in map(int, a.terms.split(",")) for r in map(float, a.rates.split(","))]
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        done = list(pool.map(lambda kr: run(a.playground.resolve(), runs, mesh.resolve(), kr[0], kr[1], a.eps,
                                            a.steps), todo))
    with open(a.out / "curves.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["terms", "rate", "step", "t", "strain", "stress"])
        for terms, rate, dt, steps, _, _, _, rows in done:
            ramp = a.eps / rate
            for i, (t, force) in enumerate(rows, 1):
                strain = a.eps * (t / ramp if t <= ramp else 2 - t / ramp)
                out.writerow([terms, "%g" % rate, i, "%.17g" % t, "%.17g" % strain, "%.17g" % force])
    with open(a.out / "runs.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["terms", "rate", "dt", "steps", "stored", "rc", "complete"])
        for terms, rate, dt, steps, stored, rc, complete, _ in done:
            out.writerow([terms, "%g" % rate, "%.6g" % dt, 2 * steps, stored, rc, int(complete)])
            print("terms %2d rate %-6g dt %-10.3g stored %-3s rc %d complete %d" % (terms, rate, dt, stored, rc,
                                                                                  complete))


if __name__ == "__main__":
    main()
