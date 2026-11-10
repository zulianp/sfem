#!/usr/bin/env python3
"""The tests of 70 EPDM 281 repeated on one element, with the valves' material.

    measured_tests.py <playground> <measurements> <out> [--terms 4,15] [--steps 2000]

<measurements> holds uniaxial.csv, equibiax.csv and pureshear.csv (t, engineering strain, engineering
stress in MPa; epdm281_extract.py). Per test, one HEX8 unit cube follows the measured strain history,
quasi-statically, the history in double, with the full series (14) and the series cut to its K slowest
terms and rescaled (stress_strain.py):
  uniaxial   x pulled; x, y, z = 0 held normal;
  equibiax   x and y pulled alike; x, y, z = 0 held normal;
  pureshear  x pulled; x = 0 and z = 0 held normal, both y faces held (the planar test's width).
The stress is the force on the pulled x face over its initial area. <out> must not exist; writes
<out>/curves.csv (test, terms, step, t, strain, stress) and <out>/runs.csv.
"""

import argparse
import concurrent.futures
import csv
import os
from pathlib import Path
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_strain import ROOT, material  # noqa: E402

TESTS = {
    "uniaxial": (["pull_x"], [("sym_x", 0), ("sym_y", 1), ("sym_z", 2)]),
    "equibiax": (["pull_x", "pull_y"], [("sym_x", 0), ("sym_y", 1), ("sym_z", 2)]),
    "pureshear": (["pull_x"], [("sym_x", 0), ("sym_y", 1), ("top_y", 1), ("sym_z", 2)]),
}
PLANES = [("pull_x", "x", "max"), ("pull_y", "y", "max"), ("top_y", "y", "max"),
          ("sym_x", "x", "min"), ("sym_y", "y", "min"), ("sym_z", "z", "min")]


def case(mesh, test, mat, t_end, steps):
    pulled, held = TESTS[test]
    used = set(pulled) | {n for n, _ in held}
    # Condition 0 is the pulled x face: its reaction is the force.
    dirichlet = [{"type": "sideset", "format": "name", "name": n, "component": "xy".index(n[-1]), "value": 1,
                  "profile": {"type": "tabulated", "path": "profile.csv"}} for n in pulled]
    dirichlet += [{"type": "sideset", "format": "name", "name": n, "component": c, "value": 0} for n, c in held]
    return {
        "mesh": {"path": str(mesh), "packed": False,
                 "sidesets": [{"name": n, "plane": {"axis": a, "at": w}} for n, a, w in PLANES if n in used]},
        "material": mat,
        "dirichlet_conditions": dirichlet,
        "regime": "quasi_static",
        "time": {"dt": t_end / steps, "steps": steps, "max_cutbacks": 8},
        "solver": {"newton": {"max_it": 40, "atol": 1.0e-6, "rtol": 1.0e-12, "verbose": False,
                              "tangent_projection": "on_failure"},
                   "linear": {"type": "cg", "preconditioner": "jacobi", "rtol": 1.0e-12, "max_it": 1000}},
        "output": {"path": "out", "export_freq": steps, "quantities": True},
    }


def run(playground, root, mesh, test, terms, history, steps, mat=None):
    """One test; `mat` replaces the series cut to `terms` (stress_strain.material) when given."""
    w = root / ("%s_k%02d" % (test, terms))
    w.mkdir()
    # The unit cube's displacement is the engineering strain.
    (w / "profile.csv").write_text("".join("%.10g, %.10g\n" % (t, e) for t, e in history))
    t_end = history[-1][0]
    (w / "case.yaml").write_text(yaml.safe_dump(case(mesh, test, mat or material(terms), t_end, steps), sort_keys=False))
    with open(w / "run.log", "w") as log:
        rc = subprocess.run([str(playground), "case.yaml", "--set", "execution_space=host"], cwd=w,
                            env=dict(os.environ, OMP_NUM_THREADS="1"), stdout=log, stderr=subprocess.STDOUT).returncode
    rows = []
    q = w / "out" / "quantities.yaml"
    if q.exists():
        for e in (yaml.safe_load(q.read_text()) or {}).get("material_objective_history") or []:
            force = [c["resultant"] for c in e["constraint_reactions"] if c["condition"] == 0][0]
            rows.append((float(e["time"]), float(force)))
    return test, terms, rc, len(rows) >= steps, rows


def strain_at(history, t):
    for (t0, e0), (t1, e1) in zip(history, history[1:]):
        if t0 <= t <= t1:
            return e0 + (e1 - e0) * (t - t0) / (t1 - t0) if t1 > t0 else e1
    return history[-1][1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("playground", type=Path)
    p.add_argument("measurements", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--terms", default="3,14")
    p.add_argument("--steps", type=int, default=2000)
    a = p.parse_args()
    if a.out.exists():
        sys.exit("measured_tests: %s exists; give a new folder" % a.out)
    a.out.mkdir(parents=True)
    mesh = a.out / "mesh"
    # box_mesh.py counts nodes per direction: two make one element.
    subprocess.run([sys.executable, str(ROOT / "python/sfem/mesh/box_mesh.py"), str(mesh), "--cell_type=HEX8",
                    "-x", "2", "-y", "2", "-z", "2", "--width", "1", "--height", "1", "--depth", "1"], check=True)
    histories = {}
    for test in TESTS:
        rows = [(float(r["t"]), float(r["strain"])) for r in csv.DictReader(open(a.measurements / (test + ".csv")))]
        histories[test] = [r for i, r in enumerate(rows) if i == 0 or r[0] > rows[i - 1][0]]
    runs = a.out / "runs"
    runs.mkdir()
    todo = [(test, k) for test in TESTS for k in map(int, a.terms.split(","))]
    with concurrent.futures.ThreadPoolExecutor(len(todo)) as pool:
        done = list(pool.map(lambda tk: run(a.playground.resolve(), runs, mesh.resolve(), tk[0], tk[1],
                                            histories[tk[0]], a.steps), todo))
    with open(a.out / "curves.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["test", "terms", "step", "t", "strain", "stress"])
        for test, terms, _, _, rows in done:
            for i, (t, force) in enumerate(rows, 1):
                out.writerow([test, terms, i, "%.10g" % t, "%.10g" % strain_at(histories[test], t), "%.10g" % (force / 1e6)])
    with open(a.out / "runs.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["test", "terms", "steps", "rc", "complete"])
        for test, terms, rc, complete, _ in done:
            out.writerow([test, terms, a.steps, rc, int(complete)])
            print("%-10s terms %2d rc %d complete %d" % (test, terms, rc, complete))


if __name__ == "__main__":
    main()
