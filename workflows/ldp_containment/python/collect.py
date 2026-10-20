#!/usr/bin/env python3
"""The benchmark's runs as CSV.

usage: collect.py <results> [<results> ...] [--out=folder]

<results> is a folder bench.sbatch.sh wrote, or a parent of several (one per job): every run
folder under it with an env.txt is a row of runs.csv. summary.csv has one row per case and
configuration (and node count): the median, minimum and maximum of each time over the repeats
that succeeded, and the problem's sizes and iteration counts, which do not depend on the repeat.

Columns of runs.csv:
  case, config, kind (host/gpu), nodes, ranks, ranks_per_node, threads, gpus, repeat, job, rc,
  case_yaml_sha                          the case file the run solved (a prefix of its sha256)
  elements, dofs, contact_nodes          the problem (global; owned elements and dofs under MPI)
  steps, newton, qp_iterations, final    pseudo-steps, Newton and MPRGP iterations over them, the
                                         last step's |r|/|r0|
  wall                                   srun to exit, start-up and output included (s)
  setup, solve, qp, linearize, assemble  Playground::create, Stepper::run, the QP solves, the
                                         linearisations, the tangent assemblies (s, rank 0's trace)
  exchange_begin, exchange_wait          the ghost gathers' two halves (s)
  qp_per_iteration                       qp / qp_iterations (us)
"""

import csv
import getopt
import os
import re
import statistics
import sys

STEP = re.compile(r"^step \d+ .*?: (\w+), (\d+) Newton iterations?, (\d+) linear, \|r\|/\|r0\| = ([0-9.eE+-]+)")
PLAYGROUND = re.compile(r"^Playground: .*?, (\d+) elements, (\d+) dofs")
CONTACT = re.compile(r"^Contact: (\d+) nodes")
TRACE = {
    "setup": "Playground::create",
    "solve": "Stepper::run",
    "qp": "NodeObstacleStep::solve",
    "linearize": "NodeObstacleStep::linearize",
    "assemble": "BSRTangent::assemble",
    "exchange_begin": "Exchange::gather_begin",
    "exchange_wait": "Exchange::gather_wait",
}
KEYS = ["case", "config", "kind", "nodes", "ranks", "ranks_per_node", "threads", "gpus", "repeat", "job", "rc",
        "case_yaml_sha"]
SIZES = ["elements", "dofs", "contact_nodes", "steps", "newton", "qp_iterations", "final"]
TIMES = ["wall"] + list(TRACE) + ["qp_per_iteration"]


def read_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                env[k] = v
    return env


def read_log(path):
    out = {"elements": "", "dofs": "", "contact_nodes": "", "steps": 0, "newton": 0, "qp_iterations": 0, "final": ""}
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            m = PLAYGROUND.match(line)
            if m:
                out["elements"], out["dofs"] = int(m.group(1)), int(m.group(2))
            m = CONTACT.match(line)
            if m:
                out["contact_nodes"] = int(m.group(1))
            m = STEP.match(line)
            if m and m.group(1) != "attempt":
                out["steps"] += 1
                out["newton"] += int(m.group(2))
                out["qp_iterations"] += int(m.group(3))
                out["final"] = float(m.group(4))
    return out


def read_trace(path):
    totals = {}
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                totals[row["name"]] = float(row["total"])
    return {k: totals.get(name, "") for k, name in TRACE.items()}


def runs(roots):
    for root in roots:
        for here, _, files in os.walk(root):
            if "env.txt" in files:
                yield here


def row(folder):
    env = read_env(os.path.join(folder, "env.txt"))
    r = {k: env.get(k, "") for k in KEYS}
    r.update(read_log(os.path.join(folder, "run.log")))
    r["wall"] = float(env["wall"]) if env.get("wall") else ""
    r.update(read_trace(os.path.join(folder, "trace.csv")))
    qp, its = r["qp"], r["qp_iterations"]
    r["qp_per_iteration"] = 1e6 * qp / its if qp != "" and its else ""
    return r


def summarise(rows):
    groups = {}
    for r in rows:
        if r["rc"] == "0":
            groups.setdefault((r["case"], r["config"], r["nodes"]), []).append(r)
    out = []
    for (case, config, nodes), rs in sorted(groups.items()):
        shas = {r["case_yaml_sha"] for r in rs}
        if len(shas) > 1:
            sys.exit("collect.py: the repeats of %s %s on %s node(s) solved different case files %s"
                     % (case, config, nodes, sorted(shas)))
        s = {k: rs[0][k] for k in KEYS if k not in ("repeat", "job", "rc")}
        s["repeats"] = len(rs)
        for k in SIZES:
            s[k] = rs[0][k]
        for k in TIMES:
            values = [r[k] for r in rs if r[k] != ""]
            for stat, f in (("median", statistics.median), ("min", min), ("max", max)):
                s["%s_%s" % (k, stat)] = f(values) if values else ""
        out.append(s)
    return out


def write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.6g" % v if isinstance(v, float) else v) for k, v in r.items() if k in fields})


def main(argv):
    opts, roots = getopt.gnu_getopt(argv[1:], "", ["out="])
    out = dict(opts).get("--out", ".")
    if not roots:
        print(__doc__)
        return 1
    rows = sorted((row(f) for f in runs(roots)), key=lambda r: (r["case"], int(r["nodes"] or 0), r["config"], int(r["repeat"] or 0)))
    os.makedirs(out, exist_ok=True)
    write(os.path.join(out, "runs.csv"), rows, KEYS + SIZES + TIMES)
    summary = summarise(rows)
    fields = [k for k in KEYS if k not in ("repeat", "job", "rc")] + ["repeats"] + SIZES
    fields += ["%s_%s" % (k, s) for k in TIMES for s in ("median", "min", "max")]
    write(os.path.join(out, "summary.csv"), summary, fields)
    failed = [r for r in rows if r["rc"] != "0"]
    print("%d runs (%d failed) -> %s" % (len(rows), len(failed), os.path.join(out, "runs.csv")))
    for r in failed:
        print("  failed: %s %s r%s (job %s, rc %s)" % (r["case"], r["config"], r["repeat"], r["job"], r["rc"]))
    print("%d configurations -> %s" % (len(summary), os.path.join(out, "summary.csv")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
