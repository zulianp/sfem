#!/usr/bin/env python3
"""Compare a CVFEM FDA-nozzle run against the benchmark's PIV data.

    python3 python/fda_nozzle_compare.py <run_out_folder> <piv_experiment_dir> [-o report.md]

<run_out_folder> is the folder the driver was given; it writes nozzle_centerline.csv (x, ux, p)
and nozzle_wall.csv (x, p) there. <piv_experiment_dir> is an unpacked `experiment/` folder
from github.com/OSEL-DAM/CFD-and-Blood-Damage-Benchmarks/tree/main/Nozzle/Data -- for the
sudden-expansion orientation this driver runs, SE_exp_0500.zip and its siblings. Each archive
holds one file per participating dataset; all of them are read and shown side by side,
because the spread between laboratories is part of what the comparison means.

Both quantities are normalised the way the benchmark papers normalise them:

    centreline velocity   u / u_in,                u_in = Q / (pi r_in^2)
    wall pressure         (p - p_ref) / (rho u_t^2 / 2),  u_t = Q / (pi r_t^2)

The pressure reference is a station both the run and the data carry, because the data's
own zero is wherever that laboratory's last tap happened to be. It is the most downstream
common station, which is where the flow has recovered and a level shift matters least.

Pure Python on purpose: the verification report is generated on machines without numpy.
This is a VALIDATION table, not a verification gate -- the published datasets disagree with
each other by several percent (Stewart et al. 2012), so nothing here is scored.
"""

import argparse
import csv
import glob
import math
import os
import sys

R_IN, R_T = 0.006, 0.002


def read_piv(path):
    """Parse one PIV_*.txt file: `key value...` header lines and `key` + count + rows blocks."""
    meta, blocks = {}, {}
    with open(path, errors="replace") as fh:
        lines = [ln.strip() for ln in fh]
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln:
            i += 1
            continue
        head = ln.split()
        key = head[0]
        if key.startswith("plot-"):
            label = " ".join(head)
            i += 1
            try:
                count = int(lines[i].split()[0])
            except (IndexError, ValueError):
                continue
            rows = []
            for k in range(count):
                parts = lines[i + 1 + k].split()
                if len(parts) >= 2:
                    rows.append((float(parts[0]), float(parts[1])))
            blocks[label] = rows
            i += 1 + count
        else:
            meta[key] = " ".join(head[1:]).strip('"')
            i += 1
    return meta, blocks


def read_csv(path, cols):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        return [tuple(float(r[c]) for c in cols) for r in rd]


def interp(xs, ys, x):
    """Linear interpolation on sorted xs; None outside the range."""
    if not xs or x < xs[0] or x > xs[-1]:
        return None
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    if xs[hi] == xs[lo]:
        return ys[lo]
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return (1 - t) * ys[lo] + t * ys[hi]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("piv")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    cl = sorted(read_csv(os.path.join(args.run, "nozzle_centerline.csv"), ("x", "ux", "p")))
    wall = sorted(read_csv(os.path.join(args.run, "nozzle_wall.csv"), ("x", "p")))
    files = sorted(glob.glob(os.path.join(args.piv, "*.txt")))
    if not files:
        sys.exit("no PIV_*.txt files in %s" % args.piv)

    out = ["# FDA nozzle: CVFEM against PIV", ""]
    meta0, _ = read_piv(files[0])
    rho = float(meta0.get("fluid-density", "1056"))
    mu = float(meta0.get("fluid-viscosity", "0.0035"))
    Q = float(meta0.get("fluid-volumetric-flow-rate", "nan"))
    u_in = Q / (math.pi * R_IN ** 2)
    u_t = Q / (math.pi * R_T ** 2)
    out += ["Orientation **%s**, throat Re %.0f (rho %g, mu %g, Q %.6g m^3/s)." % (
        meta0.get("dataset-orientation", "?"), rho * u_t * 2 * R_T / mu, rho, mu, Q),
        "Run: `%s`, %d centreline and %d wall samples." % (args.run, len(cl), len(wall)), ""]

    xs, us, ps = [r[0] for r in cl], [r[1] for r in cl], [r[2] for r in cl]
    wx, wp = [r[0] for r in wall], [r[1] for r in wall]

    datasets = []
    for f in files:
        meta, blocks = read_piv(f)
        datasets.append((meta.get("dataset-code", os.path.basename(f)), blocks))

    # ---- centreline velocity ----
    key_u = "plot-z-distribution-axial-velocity"
    stations = sorted({x for _, b in datasets for (x, _) in b.get(key_u, [])})
    hdr = ["x (m)", "CVFEM"] + ["PIV %s" % c for c, _ in datasets]
    rows = []
    for x in stations:
        row = ["%+.4f" % x]
        v = interp(xs, us, x)
        row.append("--" if v is None else "%.3f" % (v / u_in))
        for _, b in datasets:
            d = dict(b.get(key_u, []))
            row.append("%.3f" % (d[x] / u_in) if x in d else "")
        rows.append(row)
    out += ["## Centreline axial velocity, u / u_in", "", "| " + " | ".join(hdr) + " |",
            "|" + "---|" * len(hdr)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]

    # ---- wall pressure ----
    key_p = "plot-wall-distribution-pressure"
    q_dyn = 0.5 * rho * u_t ** 2
    for code, b in datasets:
        data = b.get(key_p, [])
        common = [x for (x, _) in data if interp(wx, wp, x) is not None]
        if not common:
            continue
        x_ref = max(common)
        p_run_ref = interp(wx, wp, x_ref)
        p_dat_ref = dict(data)[x_ref]
        out += ["## Wall pressure against dataset %s, (p - p(%.4f)) / (rho u_t^2 / 2)" % (code, x_ref), "",
                "| x (m) | CVFEM | PIV |", "|---|---|---|"]
        for x, p in data:
            v = interp(wx, wp, x)
            out.append("| %+.4f | %s | %.3f |" % (
                x, "--" if v is None else "%.3f" % ((v - p_run_ref) / q_dyn), (p - p_dat_ref) / q_dyn))
        out.append("")

    text = "\n".join(out) + "\n"
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
