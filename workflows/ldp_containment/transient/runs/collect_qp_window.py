#!/usr/bin/env python3
"""The QP gate's table (qp_window.sbatch): per run of $CASE/qpw_<tag>_<mesh>_<slot>_<run>, the QP's cost
per iteration and where the rest of a Newton step goes, and per tangent the A/B ratio of the medians.

    collect_qp_window.py <case> <tag> [<tag> ...]

Columns: dofs (from the output mesh), steps, failed attempts, Newton and QP iterations (history.csv),
us per QP iteration (NodeObstacleStep::solve_qp over the QP iterations), the CG runs' share, and per
Newton step the update (BSRTangent::assemble or Function::inexact_update), the scaling
(NodeObstacleStep::update_scaling) and the eigenvalue bound (NodeObstacleStep::max_eig and
::estimate_max_eig), in ms; the tangent's and the device's bytes from the memory probe. Stdlib only.
"""
import csv
import os
import re
import statistics
import sys

UPDATE = ("BSRTangent::assemble", "Function::inexact_update")


def trace(folder):
    out = {}
    path = os.path.join(folder, "smesh.trace.csv")
    if os.path.exists(path):
        for row in csv.reader(open(path)):
            if len(row) >= 3:
                try:
                    out[row[0]] = (int(row[1]), float(row[2]))
                except ValueError:
                    pass
    return out


def seconds(t, *names):
    return sum(t.get(n, (0, 0.0))[1] for n in names)


def dofs(folder):
    meta = os.path.join(folder, "sim2", "mesh", "x.float32")
    return 3 * os.path.getsize(meta) // 4 if os.path.exists(meta) else 0


def row(folder):
    rows = list(csv.DictReader(open(os.path.join(folder, "sim2", "history.csv"))))
    steps = [r for r in rows[1:] if r["status"] == "converged"]
    t = trace(folder)
    newton = sum(int(r["newton"]) for r in steps)
    its = sum(int(r["linear"]) for r in steps)
    log = open(os.path.join(folder, "run.log")).read() if os.path.exists(os.path.join(folder, "run.log")) else ""
    last = rows[-1]
    tangent = sum(float(last.get("bytes_%s_device" % c) or 0) for c in ("tangent_values", "tangent_structure", "tangent_work"))
    qp = seconds(t, "NodeObstacleStep::solve_qp")
    return {
        "dofs": dofs(folder), "steps": len(steps), "t_end": float(last["t"]), "failed": len(re.findall("attempt failed", log)),
        "newton": newton, "qp_its": its, "us_per_qp_it": 1e6 * qp / its if its else 0,
        "cg_share": seconds(t, "MPRGP::cg_run") / qp if qp else 0,
        "update_ms": 1e3 * seconds(t, *UPDATE) / newton if newton else 0,
        "scaling_ms": 1e3 * seconds(t, "NodeObstacleStep::update_scaling") / newton if newton else 0,
        "bound_ms": 1e3 * seconds(t, "NodeObstacleStep::max_eig", "NodeObstacleStep::estimate_max_eig") / newton if newton else 0,
        "qp_s": qp, "stepper_s": seconds(t, "Stepper::run"),
        "tangent_mb": tangent / 1e6, "peak_mb": max(float(r.get("device_used_peak") or 0) for r in rows) / 1e6,
        "pool_mb": max(float(r.get("device_pool_peak") or 0) for r in rows) / 1e6,
    }


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    case, tags = argv[0], argv[1:]
    pattern = re.compile(r"^qpw_(?P<tag>[^_]+)_(?P<mesh>s0|p2)_(?P<slot>[AB]\d)_(?P<run>.+)$")
    table = {}
    for name in sorted(os.listdir(case)):
        m = pattern.match(name)
        if not m or m.group("tag") not in tags:
            continue
        folder = os.path.join(case, name)
        if not os.path.exists(os.path.join(folder, "sim2", "history.csv")):
            print("%s: no history" % name)
            continue
        r = row(folder)
        table.setdefault((m.group("tag"), m.group("mesh"), m.group("run")), {})[m.group("slot")] = r
        print("%-6s %-3s %-3s %-24s dofs %8d steps %4d failed %d newton %5d qp_its %8d us/it %7.1f cg %.2f update %6.2f "
              "scaling %5.2f bound %6.2f ms/newton qp %7.1f s total %7.1f s tangent %7.1f MB peak %7.1f MB pool %7.1f MB" % (
                  m.group("tag"), m.group("mesh"), m.group("slot"), m.group("run"), r["dofs"], r["steps"], r["failed"], r["newton"],
                  r["qp_its"], r["us_per_qp_it"], r["cg_share"], r["update_ms"], r["scaling_ms"], r["bound_ms"], r["qp_s"],
                  r["stepper_s"], r["tangent_mb"], r["peak_mb"], r["pool_mb"]))
    print()
    for (tag, mesh, run), slots in sorted(table.items()):
        a = [slots[s] for s in ("A1", "A2") if s in slots]
        b = [slots[s] for s in ("B1", "B2") if s in slots]
        # Unfinished runs (no traced QP yet) are left out.
        a = [r for r in a if r["us_per_qp_it"] > 0]
        b = [r for r in b if r["us_per_qp_it"] > 0]
        if not a or not b:
            continue
        med = lambda rs, k: statistics.median(r[k] for r in rs)
        print("%-6s %-3s %-24s us/it A %7.1f B %7.1f (B/A %.3f, A spread %.3f)  total A %7.1f B %7.1f (B/A %.3f)  qp_its B/A %.3f"
              "  update ms A %6.2f B %6.2f  peak MB A %7.1f B %7.1f  pool MB A %7.1f B %7.1f" % (
            tag, mesh, run, med(a, "us_per_qp_it"), med(b, "us_per_qp_it"), med(b, "us_per_qp_it") / med(a, "us_per_qp_it"),
            max(r["us_per_qp_it"] for r in a) / min(r["us_per_qp_it"] for r in a) - 1,
            med(a, "stepper_s"), med(b, "stepper_s"), med(b, "stepper_s") / med(a, "stepper_s"),
            med(b, "qp_its") / med(a, "qp_its"), med(a, "update_ms"), med(b, "update_ms"),
            med(a, "peak_mb"), med(b, "peak_mb"), med(a, "pool_mb"), med(b, "pool_mb")))


if __name__ == "__main__":
    main(sys.argv[1:])
