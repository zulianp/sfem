#!/usr/bin/env python3
"""Turn SFEM trace CSVs into a throughput and bottleneck report.

    python3 cvfem_trace_report.py runs.tsv -o docs/CVFEM_Throughput.md --html

runs.tsv is one line per configuration:  label <TAB> path <TAB> ndof <TAB> threads <TAB> machine
and each path is a smesh.trace.csv (name,calls,total,avg) from one run.

Throughput is degrees of freedom per second, which for a routine called `calls` times on a
problem of `ndof` is calls * ndof / total. Reported in MDOF/s so it can be compared against
the numbers in perf/baseline_grace.csv and docs/CVFEM_Kernels.md.

Standard library only, for the same reason the verification report is: numpy, matplotlib
and markdown are absent from the Alps uenv and from the development laptop's python3.
"""

import argparse
import datetime
import html
import os
import shutil
import subprocess
import sys

# Scopes that CONTAIN others. Summing every row double counts, and a share taken against
# that sum understates everything -- the same trap the driver's own phase_report documents,
# where an earlier reading reported fine-level smoothing at 42% when it was 79%. There is no
# parent information in the CSV, so the nesting is declared here rather than inferred.
CONTAINERS = {
    "BiCGStab::apply", "FGMRES::apply", "ConjugateGradient::apply",
    "Function::apply", "Function::gradient", "Function::hessian_block_diag",
    # Function:: wraps the DirichletConditions:: call of the same name; both are real scopes
    # and counting both counts the same seconds twice.
    "Function::copy_constrained_dofs", "Function::constraints_gradient",
    "Function::apply_constraints", "Function::apply_zero_constraints",
    "CVFEMNavierStokes::apply", "CVFEMNavierStokes::gradient",
    "CVFEMNavierStokes::hessian_block_diag", "CVFEMNavierStokes::hessian_bsr",
    "CVFEMNavierStokes::apply_blocks", "CVFEMNavierStokes::initialize",
    "cvfem_hex8_ns_steady::apply_jacobian_action", "cvfem_hex8_ns_steady::apply_residual",
    # The accumulate half of the Jacobian action wraps the packed element sweep, the
    # direction-gradient reconstruction and the boundary closure. It used to share the name
    # above -- the wrapper and the accumulate both announced themselves as
    # apply_jacobian_action, which is why it was renamed -- and the rename left this list
    # behind, so the container came back as a leaf and every second inside it was counted
    # twice. It read 47% of a run whose children already accounted for the same seconds.
    "cvfem_hex8_ns_steady::apply_jacobian_action_accumulate",
    "cvfem_hex8_ns_steady::assemble_jacobian", "cvfem_hex8_ns_steady::assemble_nodal_p_grad",
    "cvfem_hex8_ns_steady::assemble_nodal_q_grad",
    "sscvfem::apply", "sscvfem::residual", "sscvfem::apply_blocks",
    "sscvfem::nodal_p_grad", "sscvfem::nodal_q_grad", "sscvfem::init",
    "cvfem_ss::assemble_hierarchy",
}

# What each row IS, so a reader can tell an element sweep from bookkeeping. Anything not
# listed is reported under "other", never silently dropped.
KIND = [
    ("element sweep",   ("apply_residual_", "apply_jacobian_action_", "assemble_jacobian_",
                         "apply_macro_local", "apply_naive", "residual_naive", "block_diag",
                         "apply_blocks_")),
    ("nodal gradient",  ("nodal_grad_strided",)),
    ("boundary",        ("boundary_scs",)),
    ("smoother",        ("vanka_", "block_jacobi", "galerkin_")),
    ("constraints",     ("Dirichlet", "constrained", "constraints_")),
    ("transient",       ("transient",)),
    ("setup",           ("init", "build_scatter", "node_volume", "make_bsr4", "precompute_",
                         "vanka_setup", "build_block_jacobi", "crs", "mark_constraints")),
]


def kind_of(name):
    for label, keys in KIND:
        if any(k in name for k in keys):
            return label
    return "other"


def read_trace(path):
    rows = []
    with open(path) as fh:
        head = fh.readline()
        if not head.lower().startswith("name,"):
            fh.seek(0)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 4:
                continue
            try:
                rows.append({"name": parts[0], "calls": int(parts[1]), "total": float(parts[2])})
            except ValueError:
                continue
    return rows


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def mdofs(calls, ndof, total):
    return (calls * ndof / total / 1e6) if total > 0 else 0.0


# The one element sweep each operator actually runs in the solve, so the headline table
# compares like with like rather than whichever scope happened to be slowest.
MAIN_SWEEP = ["cvfem_hex8_ns_steady::apply_jacobian_action_packed",
              "cvfem_hex8_ns_steady::apply_jacobian_action_sumfact",
              "sscvfem::apply_macro_local_hoisted"]


def section_summary(configs, traces):
    body = ["## Summary", "",
            "The element sweep that dominates the solve, in MDOF/s -- `calls * dof / seconds`",
            "on the scope the Krylov iteration actually spends its time in.", ""]
    rows = []
    for c in configs:
        r = next((x for x in traces[c["label"]] if x["name"] in MAIN_SWEEP), None)
        if not r:
            continue
        lo = [x for x in traces[c["label"]] if x["name"] not in CONTAINERS]
        tot = sum(x["total"] for x in lo)
        agg = {}
        for x in lo:
            agg[kind_of(x["name"])] = agg.get(kind_of(x["name"]), 0.0) + x["total"]
        pct = lambda k: "%.0f%%" % (100.0 * agg.get(k, 0.0) / tot) if tot else "--"
        rows.append([c["label"], "{:,}".format(c["ndof"]), r["name"].split("::")[-1],
                     "%.0f" % mdofs(r["calls"], c["ndof"], r["total"]),
                     pct("element sweep"), pct("nodal gradient"), pct("boundary"),
                     pct("constraints")])
    body.append(table(["run", "dof", "sweep", "MDOF/s", "sweep", "nodal grad", "boundary",
                       "constraints"], rows))
    body += ["The boundary closure is a separate pass on the flat operator and is FUSED into",
             "the macro-element sweep on the semi-structured one, where it is inside the micro",
             "loop and cannot carry a scope of its own without paying for one per element. So",
             "its column is blank for the semi-structured rows and its cost is inside theirs;",
             "the two boundary shares are not comparable and the sweep shares are not either.",
             ""]
    return "\n".join(body) + "\n"


def section_config(cfg, rows, top):
    """Where the time goes in one configuration, and at what throughput."""
    leaves = [r for r in rows if r["name"] not in CONTAINERS]
    leaves.sort(key=lambda r: -r["total"])
    tot = sum(r["total"] for r in leaves)

    body = ["### %s" % cfg["label"], "",
            "%s dof, %s threads, %s. %.3f s in non-container scopes." %
            ("{:,}".format(cfg["ndof"]), cfg["threads"], cfg["machine"], tot), ""]
    out = []
    for r in leaves[:top]:
        out.append([("`%s`" % r["name"]), kind_of(r["name"]), r["calls"], "%.3f" % r["total"],
                    "%.1f" % (1e6 * r["total"] / r["calls"]),
                    "%.1f" % mdofs(r["calls"], cfg["ndof"], r["total"]),
                    "%.1f%%" % (100.0 * r["total"] / tot if tot else 0)])
    body.append(table(["scope", "kind", "calls", "seconds", "us/call", "MDOF/s", "share"], out))

    # By kind, which is the answer to "where does it actually go".
    agg = {}
    for r in leaves:
        agg.setdefault(kind_of(r["name"]), [0.0, 0])
        agg[kind_of(r["name"])][0] += r["total"]
        agg[kind_of(r["name"])][1] += r["calls"]
    body.append(table(["kind", "seconds", "share"],
                      [[k, "%.3f" % v[0], "%.1f%%" % (100.0 * v[0] / tot if tot else 0)]
                       for k, v in sorted(agg.items(), key=lambda kv: -kv[1][0])]))

    cont = [r for r in rows if r["name"] in CONTAINERS]
    cont.sort(key=lambda r: -r["total"])
    if cont:
        body += ["Containers, listed apart because the rows above are inside them and adding",
                 "both would count the same seconds twice:", ""]
        body.append(table(["scope", "calls", "seconds", "us/call"],
                          [["`%s`" % r["name"], r["calls"], "%.3f" % r["total"],
                            "%.1f" % (1e6 * r["total"] / r["calls"])] for r in cont[:8]]))
    return "\n".join(body) + "\n"


def section_scaling(configs, traces, scopes):
    """One scope's throughput across sizes -- which is where saturation shows."""
    body = ["## Throughput against problem size", "",
            "MDOF/s per scope, the same scope across every configuration. A kernel that is",
            "memory bound flattens; one that is not keeps climbing with the problem until it",
            "does. A number measured below saturation is not a throughput, so the smallest",
            "sizes are here to show where that begins rather than to be quoted.", ""]
    for scope in scopes:
        rows, any_row = [], False
        for cfg in configs:
            r = next((x for x in traces[cfg["label"]] if x["name"] == scope), None)
            if not r:
                rows.append([cfg["label"], "{:,}".format(cfg["ndof"]), "--", "--", "--"])
                continue
            any_row = True
            rows.append([cfg["label"], "{:,}".format(cfg["ndof"]), r["calls"], "%.3f" % r["total"],
                         "%.1f" % mdofs(r["calls"], cfg["ndof"], r["total"])])
        if any_row:
            body += ["### `%s`" % scope, "", table(["run", "dof", "calls", "seconds", "MDOF/s"], rows)]
    return "\n".join(body) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", help="TSV: label, trace path, ndof, threads, machine")
    ap.add_argument("-o", "--output", default="CVFEM_Throughput.md")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--top", type=int, default=16, help="rows per configuration (default 16)")
    # Prose that outlives a rerun. The tables here are regenerated from scratch every time,
    # so any analysis written into the output by hand is destroyed by the next run -- which
    # is how the account of the Rhie-Chow and boundary findings would be lost the first time
    # anyone re-measured. Keep it in a file and pass it here instead.
    ap.add_argument("--prose", action="append", default=[], metavar="FILE",
                    help="markdown file inserted after the summary, before the per-configuration "
                         "tables; repeatable, inserted in the order given")
    args = ap.parse_args()

    configs = []
    with open(args.runs) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            configs.append({"label": p[0], "path": p[1], "ndof": int(p[2]),
                            "threads": p[3], "machine": p[4]})
    if not configs:
        sys.exit("no configurations in %s" % args.runs)

    traces = {}
    for c in configs:
        if not os.path.exists(c["path"]):
            sys.exit("missing trace: %s" % c["path"])
        traces[c["label"]] = read_trace(c["path"])

    doc = ["# CVFEM solver throughput and bottlenecks", "",
           "Where the time goes in the CVFEM Navier-Stokes solve, on both the flat HEX8 and",
           "the semi-structured operator, measured by the scopes compiled into the solver",
           "rather than by a sampling profiler -- so every row is a named routine and the",
           "shares add up.", "",
           "Throughput is `calls * dof / seconds`, in MDOF/s. For an element sweep that is the",
           "rate the operator processes unknowns; for a setup routine called once it is not a",
           "rate at all and is there only for scale.", "",
           ]
    doc.append(section_summary(configs, traces))
    for path in args.prose:
        with open(path) as fh:
            doc.append(fh.read().rstrip("\n"))
        doc.append("")
    doc += ["## Per configuration", ""]
    for c in configs:
        doc.append(section_config(c, traces[c["label"]], args.top))

    # The scopes worth following across sizes: the element sweeps and the two things that
    # ride along with every apply.
    follow = []
    for c in configs:
        for r in traces[c["label"]]:
            if r["name"] in CONTAINERS:
                continue
            if kind_of(r["name"]) in ("element sweep", "nodal gradient", "boundary", "smoother"):
                if r["name"] not in follow:
                    follow.append(r["name"])
    doc.append(section_scaling(configs, traces, follow))

    doc += ["## Provenance", "",
            table(["field", "value"],
                  [["generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                   ["configurations", str(len(configs))],
                   ["machines", ", ".join(sorted({c["machine"] for c in configs}))]])]

    out = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(doc))
    print("wrote %s" % out)

    if args.html:
        exe = shutil.which("markdown_py")
        target = os.path.splitext(out)[0] + ".html"
        if not exe:
            print("markdown_py not found; wrote Markdown only")
            return
        for exts in (["tables"], []):
            argv = [exe] + sum((["-x", e] for e in exts), []) + [out]
            try:
                frag = subprocess.check_output(argv, stderr=subprocess.DEVNULL).decode()
            except subprocess.CalledProcessError:
                continue
            css = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs",
                               "style.cscs.css")
            dst = os.path.join(os.path.dirname(os.path.abspath(target)), "style.cscs.css")
            if os.path.exists(css) and os.path.abspath(css) != os.path.abspath(dst):
                shutil.copyfile(css, dst)
            with open(target, "w") as fh:
                fh.write('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                         '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                         '<title>CVFEM throughput</title>\n'
                         '<link rel="stylesheet" href="style.cscs.css">\n</head>\n<body>\n'
                         '<div class="report-head"><span class="mark">CVFEM</span>'
                         '<span>throughput and bottlenecks</span></div>\n' + frag +
                         "\n</body>\n</html>\n")
            print("wrote %s" % target)
            return
        print("markdown_py failed; the Markdown stands on its own")


if __name__ == "__main__":
    main()
