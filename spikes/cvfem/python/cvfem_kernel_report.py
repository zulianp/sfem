#!/usr/bin/env python3
"""Turn CVFEM benchmark CSVs into a kernel report that says what it measured.

    python3 cvfem_kernel_report.py bench.csv -o docs/CVFEM_Kernels.md --html

Every rate in this report is shown next to the operator it was measured on. That is the
whole point of it. The benchmark can be asked for a configuration it cannot honour -- the
Jacobian action ignores --kernel entirely, the isoparametric residual on a pack-based
layout always runs the isoparametric SIMD kernel, the assembled Jacobian keeps the frozen
Rhie-Chow form on purpose -- and a table of MDOF/s with no column saying so invites the
reader to compare two different operators and call the difference a speedup.

So the report reads the `ran_*` columns, which the driver sets at the dispatch site, and
never the request columns. A row that asked for Rhie-Chow and did not get it reads
`off` here no matter what the flag said.

Standard library only, for the same reason the verification and throughput reports are:
numpy, matplotlib and markdown are absent from the Alps uenv's python3 and from the
default python3 on the development laptop.
"""

import argparse
import csv
import datetime
import os
import shutil
import subprocess
import sys


# ---------------------------------------------------------------- the operator's terms
#
# The reference is what the solver evaluates, not what the benchmark can be made to do.
# For a packed Jacobian action that is, in order: the nodal pressure gradient (cached
# across a Krylov solve), the direction's gradient (every matvec), the hoisted Rhie-Chow
# coefficient, the element sweep, the boundary closure, the transient term, constraints.
#
# Each entry is (column, heading, how to render the cell). Reading `ran_*` rather than the
# request is the reason this report exists.
TERMS = [
    ("ran_rc",       "Rhie-Chow",  lambda r: {"off": "–", "on": "carried", "exact": "carried",
                                              "frozen": "frozen by design"}.get(r.get("ran_rc", "off"), "?")),
    ("exact_rc",     "exact-RC J", lambda r: "carried" if r.get("exact_rc") == "1"
                                   else ("frozen by design" if r.get("ran_rc") == "frozen" else "–")),
    # Whether the nodal pressure gradient is part of the apply or hoisted out of the whole
    # Krylov solve. Not a decoration: rebuilding it per apply is a full element sweep, and
    # it is a stage of the cascade in its own right -- so it belongs among the terms rather
    # than as a footnote, and without it here the two rows would collapse into one and the
    # dearer of them would disappear.
    ("pgrad_per_apply", "state ∇p", lambda r: "per apply" if r.get("pgrad_per_apply") == "1"
                                    else ("hoisted" if r.get("ran_rc", "off") != "off" else "–")),
    ("ran_boundary", "boundary",   lambda r: "carried" if r.get("ran_boundary") == "on" else "–"),
    ("upwind_eps",   "upwind band", lambda r: "–" if float(r.get("upwind_eps", 0) or 0) == 0 else "carried"),
    ("transient",    "transient",  lambda r: "carried" if r.get("transient") == "1" else "–"),
]


def condition(r):
    """How the measurement was taken, as opposed to what it computed.

    Only one so far: --live-vectors keeps N extra vectors of the solution size resident and
    takes the direction from them in rotation, which is what a Krylov iteration does and
    what the default -- an apply replayed on a warm, small working set -- does not. It
    changes the number materially, so two rows that differ only in this are different
    measurements and must not reduce to their best.
    """
    n = r.get("live_vectors", "0") or "0"
    return "cold, %s live" % n if n not in ("0", "") else "warm"


def variant(r):
    """The name a reader would use for this row's code path, not its request."""
    k = r.get("ran_kernel", "") or "?"
    parts = [r["layout"], k if k != "n/a" else "(kernel n/a)"]
    if r.get("geom") and r["geom"] != "affine":
        parts.append(r["geom"])
    return " / ".join(parts)


def completeness(r):
    """One word for how far this row is from the operator the solver runs."""
    rc = r.get("ran_rc", "off")
    bnd = r.get("ran_boundary", "off") == "on"
    if rc in ("on", "exact") and bnd:
        return "solver operator"
    if rc in ("on", "exact") or bnd:
        return "partial"
    if rc == "frozen":
        return "frozen-RC"
    return "element kernel"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def read_rows(paths):
    """Best rate per (operation, variant, coverage, dof).

    Best rather than mean: on a shared machine background load can only make a run slower,
    so the maximum is the estimator that converges on the truth as runs accumulate. Rows
    from a build that predates the ran_* columns are dropped rather than guessed at --
    that is exactly the misattribution this report exists to prevent.
    """
    best, skipped = {}, 0
    for p in paths:
        with open(p) as fh:
            for raw in csv.DictReader(fh):
                if "ran_kernel" not in raw or raw.get("ran_kernel") in (None, ""):
                    skipped += 1
                    continue
                try:
                    rate = float(raw["MDOF_s"])
                    raw["dofs"] = int(raw["dofs"])
                except (KeyError, TypeError, ValueError):
                    skipped += 1
                    continue
                key = (raw["operation"], variant(raw), condition(raw),
                       tuple(f(raw) for _, _, f in TERMS), raw["dofs"])
                if key not in best or rate > float(best[key]["MDOF_s"]):
                    best[key] = raw
    return list(best.values()), skipped


def section_coverage(rows):
    """What each variant computes. First, because it is what makes the rest readable."""
    # Keyed on the terms as well as the variant, not on the variant alone. The same code
    # path measured with and without Rhie-Chow is two different operators and this table's
    # whole subject is which terms ran -- collapsing them showed one row and hid the rest,
    # so a block diagonal measured four ways appeared here once.
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: (r["operation"], variant(r),
                                         tuple(f(r) for _, _, f in TERMS))):
        key = (r["operation"], variant(r), tuple(f(r) for _, _, f in TERMS))
        if key in seen:
            continue
        seen.add(key)
        out.append([r["operation"], variant(r)] + [f(r) for _, _, f in TERMS] + [completeness(r)])
    if not out:
        return ""
    return ("## What each variant computes\n\n"
            "Read from the `ran_*` columns, which the benchmark sets where it dispatches --\n"
            "not from the flags it was passed. A variant that asked for a term and did not\n"
            "get it shows a dash here. `frozen by design` is the assembled Jacobian, which\n"
            "keeps the frozen Rhie-Chow form deliberately: the exact term couples pressures\n"
            "beyond nearest neighbours and would widen the BSR pattern, and that operator\n"
            "exists only to build the preconditioner.\n\n"
            + table(["operation", "variant"] + [h for _, h, _ in TERMS] + ["completeness"], out))


def section_throughput(rows):
    """The rates, each beside the operator it was measured on."""
    out = []
    for r in sorted(rows, key=lambda r: (r["operation"], -float(r["MDOF_s"]))):
        out.append([r["operation"], variant(r), "{:,}".format(r["dofs"]),
                    "%.0f" % float(r["MDOF_s"]), completeness(r), condition(r), r.get("threads", "?")])
    if not out:
        return ""
    return ("## Throughput\n\n"
            "Best observed rate per configuration, in MDOF/s. Best rather than mean because\n"
            "background load can only make a run slower. The completeness column is the one\n"
            "from the table above: two rows are comparable only when it matches.\n\n"
            + table(["operation", "variant", "dof", "MDOF/s", "completeness", "working set", "threads"], out))


def section_cascade(rows):
    """What each term costs, on one layout, at one size, for one operation at a time."""
    body = []
    for op in ("residual", "jac_action"):
        # One variant, not one layout. A cascade is a table in which only the OPERATOR
        # varies, so mixing kernels or geometries into it puts two effects in one column --
        # which it did: the bare `element kernel` row appeared eight times, once per member
        # of the variant sweep, and none of them was a stage of anything.
        pool = [r for r in rows
                if r["operation"] == op and r["layout"] == "packed"
                and r.get("geom", "affine") == "affine"
                and r.get("ran_kernel") in ("sumfact", "n/a")]
        if not pool:
            continue
        # One size only: a cascade across sizes is two effects at once.
        size = max({r["dofs"] for r in pool}, key=lambda d: len([r for r in pool if r["dofs"] == d]))
        pool = [r for r in pool if r["dofs"] == size]
        base = [r for r in pool if completeness(r) == "element kernel"]
        if not base:
            continue
        ref = max(float(r["MDOF_s"]) for r in base)
        out = []
        for r in sorted(pool, key=lambda r: -float(r["MDOF_s"])):
            rate = float(r["MDOF_s"])
            terms = ", ".join("%s (%s)" % (h, f(r)) if f(r) not in ("carried",) else h
                              for _, h, f in TERMS if f(r) not in ("–", "?"))
            out.append([completeness(r), terms or "none", condition(r), "%.0f" % rate,
                        "%.2fx" % (ref / rate) if rate > 0 else "–"])
        body.append("### %s, %s dof\n\n" % (op, "{:,}".format(size)) +
                    table(["operator", "terms carried", "working set", "MDOF/s",
                           "cost vs bare kernel"], out))
    if not body:
        return ""
    return ("## What the physics costs\n\n"
            "The element kernel alone against the same kernel carrying what the solver\n"
            "needs, one operation and one size at a time so that only the operator varies.\n"
            "The bare-kernel row is the number the regression gate tracks and the one every\n"
            "quoted figure has historically meant; the solver does not run it.\n\n"
            + "\n".join(body))


def section_provenance(rows, paths, skipped):
    hosts = sorted({r.get("host", "?") for r in rows})
    facts = [["generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
             ["rows", str(len(rows))],
             ["machines", ", ".join(hosts) if hosts else "?"],
             ["sources", ", ".join(os.path.basename(p) for p in paths)]]
    if skipped:
        facts.append(["rows skipped (no ran_* columns)", str(skipped)])
    return ("## Provenance\n\n" + table(["field", "value"], facts) +
            "\nRegenerate with `python3 python/cvfem_kernel_report.py <csv> "
            "-o docs/CVFEM_Kernels.md --html`.\n")


def build(rows, paths, skipped, prose):
    doc = ["# CVFEM kernel variants and what they measure", "",
           "Throughput of the CVFEM element kernels, with the operator each number was",
           "measured on shown beside it.",
           "",
           "The benchmark measures kernels in isolation, which is a deliberate and useful",
           "thing to do -- it is how a change to the arithmetic is seen without the rest of",
           "the solve moving underneath it. It is not the operator the Newton loop",
           "evaluates. Both appear here, labelled, because the difference between them has",
           "been mistaken for a speedup before.",
           ""]
    for s in (section_coverage(rows), section_throughput(rows), section_cascade(rows)):
        if s.strip():
            doc.append(s)
            doc.append("")
    for p in prose:
        with open(p) as fh:
            doc.append(fh.read().rstrip("\n"))
        doc.append("")
    doc.append(section_provenance(rows, paths, skipped))
    return "\n".join(doc)


def write_html(md_path, title="CVFEM kernels"):
    exe = shutil.which("markdown_py") or shutil.which("markdown_py3")
    target = os.path.splitext(md_path)[0] + ".html"
    if not exe:
        print("markdown_py not found; wrote Markdown only")
        return
    for exts in (["tables"], []):
        argv = [exe] + sum((["-x", e] for e in exts), []) + [md_path]
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
                     '<title>%s</title>\n'
                     '<link rel="stylesheet" href="style.cscs.css">\n</head>\n<body>\n'
                     '<div class="report-head"><span class="mark">CVFEM</span>'
                     '<span>kernel variants</span></div>\n' % title + frag +
                     "\n</body>\n</html>\n")
        print("wrote %s" % target)
        return
    print("markdown_py failed; the Markdown stands on its own")


def selftest():
    """The classifier and the reader, which are the parts that can quietly mislabel.

    A report that prints a wrong rate is caught by anyone who knows the kernel. A report
    that prints the right rate under the wrong operator is not, and that is the failure
    this whole file exists to prevent -- so the mapping is tested rather than trusted.
    """
    import tempfile
    bad = 0

    def check(ok, what):
        nonlocal bad
        print("%-58s %s" % (what, "OK" if ok else "FAIL"))
        if not ok:
            bad += 1

    def row(**kw):
        r = {"operation": "jac_action", "layout": "packed", "geom": "affine",
             "ran_kernel": "n/a", "ran_rc": "off", "ran_boundary": "off",
             "exact_rc": "0", "upwind_eps": "0", "transient": "0",
             "MDOF_s": "100", "dofs": "1000", "threads": "72", "host": "h"}
        r.update(kw)
        return r

    check(completeness(row()) == "element kernel", "bare kernel is not called complete")
    check(completeness(row(ran_rc="exact")) == "partial", "Rhie-Chow alone is partial")
    check(completeness(row(ran_boundary="on")) == "partial", "boundary alone is partial")
    check(completeness(row(ran_rc="exact", ran_boundary="on")) == "solver operator",
          "Rhie-Chow plus boundary is the solver operator")
    check(completeness(row(ran_rc="frozen")) == "frozen-RC", "assembled Jacobian reads frozen-RC")

    # The request must never reach the report: a row that asked for Rhie-Chow and did not
    # get it has to read as absent.
    lying = row(rhie_chow="1", ran_rc="off")
    check(TERMS[0][2](lying) == "–", "a dropped term reads absent however the flag was set")
    check(TERMS[0][2](row(ran_rc="frozen")) == "frozen by design", "frozen Rhie-Chow is labelled")

    check(variant(row(ran_kernel="n/a")) == "packed / (kernel n/a)", "an ignored kernel is named so")
    check(variant(row(ran_kernel="sumfact", layout="atomic")) == "atomic / sumfact", "variant names the code path")
    check(variant(row(ran_kernel="isoparam_simd", geom="isoparam")) ==
          "packed / isoparam_simd / isoparam", "geometry appears when it is not affine")

    # A CSV from a build without the ran_* columns must be skipped, not guessed at.
    with tempfile.TemporaryDirectory() as d:
        old = os.path.join(d, "old.csv")
        with open(old, "w") as fh:
            fh.write("operation,layout,kernel,geom,dofs,MDOF_s\nresidual,packed,sumfact,affine,1000,100\n")
        rows, skipped = read_rows([old])
        check(rows == [] and skipped == 1, "a pre-coverage CSV is skipped rather than guessed")

        new = os.path.join(d, "new.csv")
        hdr = "operation,layout,kernel,geom,dofs,MDOF_s,threads,host,ran_kernel,ran_rc,ran_boundary,exact_rc,upwind_eps,transient"
        with open(new, "w") as fh:
            fh.write(hdr + "\n")
            fh.write("residual,packed,sumfact,affine,1000,100,72,h,sumfact,off,off,0,0,0\n")
            fh.write("residual,packed,sumfact,affine,1000,140,72,h,sumfact,off,off,0,0,0\n")
        rows, skipped = read_rows([new])
        check(len(rows) == 1 and float(rows[0]["MDOF_s"]) == 140.0,
              "repeat runs of one configuration reduce to the best")

        rows, _ = read_rows([new])
        check("## What each variant computes" in build(rows, [new], 0, []),
              "the coverage section leads the report")

    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", help="benchmark CSVs written by --csv")
    ap.add_argument("-o", "--output", default="CVFEM_Kernels.md")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    # Prose that outlives a rerun, the same mechanism cvfem_trace_report.py has: the tables
    # are regenerated from scratch every time, so analysis written into the output by hand
    # is destroyed by the next measurement.
    ap.add_argument("--prose", action="append", default=[], metavar="FILE",
                    help="markdown file appended after the tables; repeatable")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.csv:
        sys.exit("no CSV given (try --help)")

    rows, skipped = read_rows(args.csv)
    if not rows:
        sys.exit("no rows with coverage columns in %s -- rebuild the benchmark and re-run"
                 % ", ".join(args.csv))
    out = args.output
    if os.path.dirname(os.path.abspath(out)):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(build(rows, args.csv, skipped, args.prose))
    print("wrote %s (%d rows, %d skipped)" % (out, len(rows), skipped))
    if args.html:
        write_html(out)


if __name__ == "__main__":
    main()
