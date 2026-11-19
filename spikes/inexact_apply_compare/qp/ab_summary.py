#!/usr/bin/env python3
"""The A/B logs of ab_library{,_gpu}.sbatch as one table: per element and store, the library's time over
the spike's (the sums over the A B B A legs: spike, spike2 against lib, lib2) for the Newton solve, the
stored tangent's updates and applies and the CG, with the legs' Newton and CG counts; with --against,
the same ratios of an earlier A/B beside them (matched by element and mesh), and the change.

    ab_summary.py <A/B log> ... [--against <earlier A/B log> ...]

stdlib only. A leg's times are its trace totals (smesh.trace.csv rows the script prints as
"  trace <scope>,<calls>,<total s>,<mean s>"); its counts are its steps' "N Newton iterations, M linear".
The GPU script prints no steps: there the Newton count is NewtonSolver::step's calls, the CG count is
unknown (-), and the update and apply ratios are left out (the device runs them asynchronously, so
their traced times are launch times; ab_library_gpu.sbatch compares Newton's totals).
"""
import re
import sys

TIMES = {"solve": "NewtonSolver::solve", "update": "Function::inexact_update", "apply": "Function::inexact_apply",
         "cg": "ConjugateGradient::apply"}
STEP = re.compile(r"converged, (\d+) Newton iterations, (\d+) linear")


def read(path):
    """{(element, mesh): {label: {"solve": s, ..., "newton": n, "linear": m}}}"""
    key, legs, leg = None, {}, None
    for line in open(path, errors="replace"):
        if line.startswith("node "):
            w = line.split()
            mesh = next((x for x in w if x.startswith("mesh.")), "?")
            key = (w[2], mesh.split("/")[-1] if "path=" in mesh else mesh)
        elif line.startswith("== "):
            leg = legs.setdefault(line.split()[1], {"newton": 0, "linear": 0})
        elif leg is not None:
            m = STEP.search(line)
            if m:
                leg["newton"] += int(m.group(1))
                leg["linear"] += int(m.group(2))
            elif line.startswith("  trace "):
                scope, calls, total = line[8:].strip().split(",")[:3]
                for k, v in TIMES.items():
                    if scope == v:
                        leg[k] = float(total)
                if scope == "NewtonSolver::step":
                    leg["steps"] = int(calls)
    return {key: legs} if key else {}


def ratios(legs):
    """{store: {"solve": lib/spike, ..., "counts": (spike newton/linear, lib newton/linear)}}"""
    out = {}
    for label in legs:
        if not label.endswith("-spike"):
            continue
        q = label[:-len("-spike")]
        a = [legs.get(q + "-spike"), legs.get(q + "-spike2")]
        b = [legs.get(q + "-lib"), legs.get(q + "-lib2")]
        if None in a or None in b:
            continue
        r = {}
        device = not a[0]["linear"]
        for k in TIMES:
            sa, sb = sum(x.get(k, 0) for x in a), sum(x.get(k, 0) for x in b)
            r[k] = sb / sa if sa and sb and not (device and k in ("update", "apply")) else None
        c = lambda x: "%d/%s" % (x.get("steps", 0), "-") if device else "%d/%d" % (x["newton"], x["linear"])
        r["counts"] = (c(a[0]), c(b[0]))
        out[q] = r
    return out


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    new, old = args, []
    if "--against" in args:
        i = args.index("--against")
        new, old = args[:i], args[i + 1:]
    olds = {}
    for p in old:
        olds.update(read(p))
    f = lambda x: "%5.3f" % x if x is not None else "  -  "
    for p in new:
        for key, legs in read(p).items():
            print("== %s %s (%s)%s" % (key[0], key[1], p.split("/")[-1], "" if key in olds else "; no earlier A/B"))
            prev = ratios(olds[key]) if key in olds else {}
            print("%-8s %-31s %-31s %-23s %s" % ("store", "lib/spike solve update apply cg", "earlier (2026-10-31)",
                                                  "solve change", "Newton/CG spike, lib"))
            for q, r in ratios(legs).items():
                o = prev.get(q, {})
                change = r["solve"] / o["solve"] if r.get("solve") and o.get("solve") else None
                print("%-8s %s %s %s %s   %s %s %s %s   %-23s %s, %s" % (
                    q, f(r["solve"]), f(r["update"]), f(r["apply"]), f(r["cg"]),
                    f(o.get("solve")), f(o.get("update")), f(o.get("apply")), f(o.get("cg")),
                    ("%+.1f%%" % (100 * (change - 1))) if change else "-", r["counts"][0], r["counts"][1]))


if __name__ == "__main__":
    main()
