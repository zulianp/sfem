#!/usr/bin/env python3
"""Apply the survival rule to the repeated CPU sweep.

A micro-kernel survives iff it is the fastest in at least one (layout, geometry) cell by
more than run-to-run noise. Below that margin it is a tie, and ties go to the simpler
kernel. Medians over the trials, so a single bad sample cannot decide anything.
"""
import csv, statistics, sys
from collections import defaultdict

NOISE = 0.04  # ~4% run-to-run on this machine, per README_alps.md
rows = list(csv.DictReader(open(sys.argv[1])))
asm = [r for r in rows if "assemble" in r["operation"]]

med = defaultdict(list)
for r in asm:
    med[(r["geom"], r["layout"], r["kernel"])].append(float(r["MDOF_s"]))
med = {k: statistics.median(v) for k, v in med.items() if v}

geoms   = sorted({k[0] for k in med})
layouts = ["atomic", "packed", "colored", "store"]
kernels = ["current", "sumfact", "sympy", "sympy_block", "sympy_row", "sympy_face"]

wins = defaultdict(list)
for g in geoms:
    print(f"\n== {g}  (median of {len(asm)//max(1,len(med))} trials, MDOF/s)")
    print("%-13s" % "kernel" + "".join("%10s" % l for l in layouts))
    best = {}
    for l in layouts:
        vals = {k: med[(g, l, k)] for k in kernels if (g, l, k) in med}
        if vals:
            best[l] = max(vals, key=vals.get)
    for k in kernels:
        line = "%-13s" % k
        for l in layouts:
            v = med.get((g, l, k))
            line += "%10s" % ("%.1f" % v if v is not None else "-")
            if v is not None and best.get(l) == k:
                bv = med[(g, l, best[l])]
                runner = sorted((med[(g, l, kk)] for kk in kernels if (g, l, kk) in med), reverse=True)
                margin = (bv - runner[1]) / runner[1] if len(runner) > 1 else 1.0
                wins[k].append((g, l, bv, margin))
        print(line)
    for l in layouts:
        if l in best:
            print(f"   best {l:8s}: {best[l]}")

print("\n== survival")
for k in kernels:
    w = wins.get(k, [])
    clear = [x for x in w if x[3] > NOISE]
    if clear:
        g, l, v, m = clear[0]
        print(f"  KEEP    {k:13s} fastest in {g}/{l} at {v:.1f} (+{100*m:.1f}% over next)")
    elif w:
        g, l, v, m = w[0]
        print(f"  TIE     {k:13s} nominally fastest in {g}/{l} at {v:.1f} but only +{100*m:.1f}% -- within noise")
    else:
        print(f"  REMOVE  {k:13s} never fastest in any cell")
