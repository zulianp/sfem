#!/usr/bin/env python3
"""The Prony history's storage formats, from the runs, into the paper's data files.

    collect_prony.py runs  <out.csv> <series.csv> <set> <case> <reference> <run> [<run> ...] [--sub sim2] [--prefix ...]
    collect_prony.py tangent <out.csv> <mesh folder>

`runs`: each run is a folder <case>/<prefix><run>/ holding <sub>/history.csv, <sub>/out/disp.<c>.<frame>.float64
(frame k being history row k), run.log and smesh.trace.csv. Its name says its storage:

    <format>[_a|_b][_p<elements per mean>][_s32]      e.g. full_a, int8_md_p16, float32_s32

(_a/_b repeats, _p the mean's run of one-point elements, _s32 float32 stress slots). Every run is
compared with <reference> at common times, each interpolated linearly between its own frames,
since each takes its own adaptive steps. Columns of <out.csv>, one row per run, appended under the
label <set>:

  e_u_max, e_u_median   100 |u - u_ref|_2 / |u_ref|_2 over the sample times [%]: the max, and the
                        median, which a valve event shifted by a step does not dominate
  E_inf                 max over the samples of max |u - u_ref| [m]
  e_inf_median, e_inf_min, e_inf_max   100 |u - u_ref|_inf / |u_ref|_inf over the sample times [%]
  cp                    max over time of the control points' |u - u_ref| over max |u_ref| of that point
  event_shift_ms, peak_rel   the valve events of ldp_figures.py (LIFT_OPEN, STROKE), if the run has valves
  bytes_*               the Prony probe's last row: H_i, D, frozen history, stress slots (all ranks)
  stepper_s             the traced Stepper::run
  updates, updates_lost the diagnostics build's counts, when present

<series.csv> gets e_u(t) and the relative infinity norm e_inf(t) per run and sample time.

`tangent`: the bytes of the float32 BSR tangent (3x3 blocks, int32 column indices and row pointers)
and of the mesh (int32 connectivity, float32 coordinates) of a mesh folder, from its node graph.
"""
import csv
import os
import re
import sys

import numpy as np

STROKE = 0.045
LIFT_OPEN = 50e-6
VALVES = [("pv_centre_uy", 1, "Pv"), ("sv_centre_uy", -1, "Sv")]
SAMPLES = 72


def parse_run(name):
    """format, repeat, elements per mean, stress slots and remainder form of a run named
    <format>[_<repeat>][_p<elements per mean>][_plain][_s32]: _plain without the remainder form,
    which every reduced format has otherwise, _s32 with float32 stress slots."""
    stress = "float32" if name.endswith("_s32") else "full"
    base = name[:-4] if name.endswith("_s32") else name
    plain = base.endswith("_plain")
    base = base[:-6] if plain else base
    m = re.match(r"^(.*?)(?:_([ab]))?(?:_p(\d+))?$", base)
    fmt, repeat, per_mean = m.group(1), m.group(2) or "a", int(m.group(3) or 0)
    return fmt, repeat, per_mean, stress, "no" if plain or fmt == "full" else "yes"


def history(run, sub):
    with open(os.path.join(run, sub, "history.csv")) as f:
        return list(csv.DictReader(f))


def frame(run, sub, k):
    return np.stack([np.fromfile(os.path.join(run, sub, "out", "disp.%d.%09d.float64" % (c, k))) for c in range(3)], axis=1)


def at_time(run, sub, times, t):
    k = min(max(int(np.searchsorted(times, t)), 1), len(times) - 1)
    t0, t1 = times[k - 1], times[k]
    w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return (1 - w) * frame(run, sub, k - 1) + w * frame(run, sub, k)


def valve_events(rows, key, sign):
    rest = float(rows[0][key])
    intervals, start = [], None
    for r in rows:
        lift, t = sign * (float(r[key]) - rest), float(r["t"])
        if start is None and lift > LIFT_OPEN:
            start = t
        elif start is not None and lift <= LIFT_OPEN:
            intervals.append((start, t))
            start = None
    t_end = float(rows[-1]["t"])
    if start is not None:
        intervals.append((start, t_end))
    strokes = []
    for k in range(int(round(t_end / STROKE))):
        lo, hi = k * STROKE, (k + 1) * STROKE
        mine = [iv for iv in intervals if lo <= iv[0] < hi]
        if not mine:
            strokes.append(None)
            continue
        main = max(mine, key=lambda iv: iv[1] - iv[0])
        peak = max(sign * (float(r[key]) - rest) for r in rows if lo <= float(r["t"]) < hi)
        strokes.append({"open": main[0], "close": main[1], "peak": peak})
    return strokes


def traced(run, name):
    path = os.path.join(run, "smesh.trace.csv")
    if not os.path.exists(path):
        return ""
    for row in csv.reader(open(path)):
        if row and row[0] == name:
            return float(row[2])
    return ""


def runs(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("series")
    ap.add_argument("set")
    ap.add_argument("case")
    ap.add_argument("reference")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--sub", default="sim2")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args(argv)

    def path(r):
        return os.path.join(a.case, a.prefix + r)

    ref = path(a.reference)
    ref_rows = history(ref, a.sub)
    ref_t = np.array([float(r["t"]) for r in ref_rows])
    samples = np.linspace(ref_t[-1] / SAMPLES, ref_t[-1], SAMPLES)
    ref_frames = {s: at_time(ref, a.sub, ref_t, s) for s in samples}
    cps = sorted({k[:-3] for k in ref_rows[0] if k.endswith("_ux")})
    valves = [v for v in VALVES if v[0] in ref_rows[0]]
    ref_events = {key: valve_events(ref_rows, col, sign) for col, sign, key in valves}
    n_dofs = 3 * ref_frames[samples[0]].shape[0]

    out, series = [], []
    for r in a.runs:
        run = path(r)
        rows = history(run, a.sub)
        t = np.array([float(x["t"]) for x in rows])
        e_us, e_infs, e_inf = [], [], 0.0
        for s in samples:
            d = at_time(run, a.sub, t, s) - ref_frames[s]
            e = 100 * np.linalg.norm(d) / np.linalg.norm(ref_frames[s])
            ei = 100 * float(np.abs(d).max()) / float(np.abs(ref_frames[s]).max())
            e_us.append(e)
            e_infs.append(ei)
            e_inf = max(e_inf, float(np.abs(d).max()))
            series.append({"set": a.set, "run": r, "t": "%.6g" % s, "e_u": "%.4e" % e, "e_inf": "%.4e" % ei})
        cp = 0.0
        for p in cps:
            for c in ("ux", "uy", "uz"):
                col = p + "_" + c
                v_ref = np.array([float(x[col]) for x in ref_rows])
                v = np.interp(ref_t, t, np.array([float(x[col]) for x in rows]))
                cp = max(cp, float(np.abs(v - v_ref).max() / max(np.abs(v_ref).max(), 1e-30)))
        shift, peak = 0.0, 0.0
        for col, sign, key in valves:
            for x, y in zip(valve_events(rows, col, sign), ref_events[key]):
                if x is None or y is None:
                    shift = float("inf")
                    continue
                shift = max(shift, abs(x["open"] - y["open"]), abs(x["close"] - y["close"]))
                peak = max(peak, abs(x["peak"] - y["peak"]) / y["peak"])
        fmt, repeat, per_mean, stress, remainder = parse_run(r)
        last = rows[-1]
        log = os.path.join(run, "run.log")
        row = {
            "set": a.set,
            "run": r,
            "format": fmt,
            "repeat": repeat,
            "elements_per_mean": per_mean,
            "stress_storage": stress,
            "remainder": remainder,
            "dofs": n_dofs,
            "steps": int(last["step"]),
            "failed": sum("attempt failed" in line for line in open(log)) if os.path.exists(log) else "",
            "stepper_s": traced(run, "Stepper::run"),
            "e_u_max": "%.4e" % max(e_us),
            "e_u_median": "%.4e" % float(np.median(e_us)),
            "E_inf": "%.4e" % e_inf,
            "e_inf_median": "%.4e" % float(np.median(e_infs)),
            "e_inf_min": "%.4e" % min(e_infs),
            "e_inf_max": "%.4e" % max(e_infs),
            "cp": "%.4e" % cp,
            "event_shift_ms": "%.4g" % (1e3 * shift) if valves else "",
            "peak_rel": "%.4e" % peak if valves else "",
        }
        for k in ("iso_stored", "bytes_history", "bytes_reference", "bytes_frozen", "bytes_stress", "host_peak_rss",
                  "device_used_peak", "updates", "updates_lost"):
            row[k] = int(float(last[k])) if k in last and last[k] != "" else ""
        out.append(row)
        print("%-8s %-18s e_u median %s max %s  bytes H %s R %s F %s S %s" % (
            a.set, r, row["e_u_median"], row["e_u_max"], row["bytes_history"], row["bytes_reference"],
            row["bytes_frozen"], row["bytes_stress"]))

    for target, rows in ((a.out, out), (a.series, series)):
        exists = os.path.exists(target)
        with open(target, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if not exists:
                w.writeheader()
            w.writerows(rows)


def read_mesh(folder):
    """Connectivity of every block and the node count, from a smesh meta.yaml (no yaml module)."""
    meta = open(os.path.join(folder, "meta.yaml")).read().splitlines()
    files, n_nodes = [], None
    for line in meta:
        m = re.match(r"\s*- i\d+: (\S+)", line)
        if m:
            files.append(m.group(1))
        m = re.match(r"n_nodes: (\d+)", line)
        if m:
            n_nodes = int(m.group(1))
    by_block = {}
    for f in files:
        by_block.setdefault(os.path.dirname(f), []).append(np.fromfile(os.path.join(folder, f), dtype=np.int32))
    return [np.stack(v, axis=1) for v in by_block.values()], n_nodes


def tangent(argv):
    out, folder = argv
    blocks, n_nodes = read_mesh(folder)
    pairs = []
    for e in blocks:
        nv = e.shape[1]
        for a in range(nv):
            for b in range(nv):
                pairs.append(e[:, a].astype(np.int64) * n_nodes + e[:, b])
    nnz_blocks = np.unique(np.concatenate(pairs)).size
    n_elements = sum(e.shape[0] for e in blocks)
    row = {
        "elements": n_elements,
        "nodes": n_nodes,
        "dofs": 3 * n_nodes,
        "nnz_blocks": nnz_blocks,
        "tangent_f32_bytes": nnz_blocks * (9 * 4 + 4) + (n_nodes + 1) * 4,
        "tangent_f64_bytes": nnz_blocks * (9 * 8 + 4) + (n_nodes + 1) * 4,
        "mesh_bytes": sum(e.size for e in blocks) * 4 + 3 * n_nodes * 4,
        "vector_bytes": 3 * n_nodes * 8,
    }
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    print(row)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("runs", "tangent"):
        sys.exit(__doc__)
    {"runs": runs, "tangent": tangent}[sys.argv[1]](sys.argv[2:])
