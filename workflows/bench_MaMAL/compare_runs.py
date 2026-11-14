#!/usr/bin/env python3
"""test_contact of two bench_MaMAL runs side by side: time, ratio, outer iterations and MG cycles.

    compare_runs.py <run A> <run B> [--labels A B] [--from-logs] [--csv out.csv]

A run is a bench_MaMAL output folder (bench_MaMAL.csv, logs/). --from-logs takes the iterations from
each log's first block of "[lagr++ k]" lines (test_contact runs first), for runs made before
SFEM_MGSDF_CONTACT_ONLY, whose stats files hold the binary's last solve.
"""
import argparse
import csv
import os
import re

LAGR = re.compile(r"^(\d+)\|(\d+)\|(\d+)\) \[lagr\+\+ \d+\].*\|\|r/r0\|\| ([0-9.e+-]+)")


def from_log(path):
    last, last_line = None, None
    for n, line in enumerate(open(path, errors="replace")):
        m = LAGR.match(line)
        if m:
            if last_line is not None and n - last_line > 5:
                break
            last, last_line = m, n
        elif last_line is not None and n - last_line > 5:
            break
    return (int(last.group(1)), int(last.group(2))) if last else (None, None)


def read(run, logs):
    out = {}
    for r in csv.DictReader(open(os.path.join(run, "bench_MaMAL.csv"))):
        key = (r["scenario"], r["line_search"], r["nl_obstacle"], int(r["base_resolution"]), int(r["element_refine_level"]))
        if logs:
            it, cyc = from_log(os.path.join(run, "logs", os.path.basename(r["log_file"])))
        else:
            it = int(r["count_iter"]) if r["count_iter"] else None
            cyc = int(r["count_mg_cycles"]) if r["count_mg_cycles"] else None
        t = float(r["test_contact_s"]) if r["test_contact_s"] else float("nan")
        out[key] = dict(nodes=int(r["nodes"]), status=r["status"], t=t, it=it, cyc=cyc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--from-logs", action="store_true")
    ap.add_argument("--csv")
    args = ap.parse_args()
    A, B = read(args.a, args.from_logs), read(args.b, args.from_logs)
    la, lb = args.labels
    w = csv.writer(open(args.csv, "w")) if args.csv else None
    if w:
        w.writerow(["scenario", "line_search", "nl_obstacle", "base_resolution", "refine", "dofs", f"{la}_s", f"{lb}_s",
                    f"{lb}_over_{la}", f"{la}_iter", f"{la}_cycles", f"{lb}_iter", f"{lb}_cycles", f"{la}_status", f"{lb}_status"])
    print(f"{'scenario':<12}{'ls':>3}{'res':>4}{'Mdofs':>8} {la + ' s':>10} {lb + ' s':>10} {'ratio':>6}  {la + ' it/cyc':>12} {lb + ' it/cyc':>12}")
    for k in sorted(set(A) & set(B), key=lambda k: (k[0], k[3], k[1] != "1")):
        a, b = A[k], B[k]
        dofs = 3 * a["nodes"]
        ratio = b["t"] / a["t"] if a["t"] else float("nan")
        print(f"{k[0]:<12}{k[1]:>3}{k[3]:>4}{dofs / 1e6:>8.1f} {a['t']:>10.2f} {b['t']:>10.2f} {ratio:>6.2f}  "
              f"{str(a['it']) + '/' + str(a['cyc']):>12} {str(b['it']) + '/' + str(b['cyc']):>12}")
        if w:
            w.writerow([k[0], k[1], k[2], k[3], k[4], dofs, a["t"], b["t"], ratio, a["it"], a["cyc"], b["it"], b["cyc"], a["status"], b["status"]])


if __name__ == "__main__":
    main()
