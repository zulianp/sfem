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
    # ran_layout, not layout, for the same reason ran_kernel is used rather than kernel: the
    # request is not what ran. --layout store has no matrix-free sweep of its own and falls
    # through to the packed one, so keying on the request put one call site in the table twice
    # under two names and invited the 0.4% between them to be read as a difference.
    lay = r.get("ran_layout") or r["layout"]
    parts = [lay, k if k != "n/a" else "(kernel n/a)"]
    if r.get("geom") and r["geom"] != "affine":
        parts.append(r["geom"])
    # The SpMV's value storage is part of the code path, not a decoration: an f32 and an f64
    # run of the same matrix are the same arithmetic over half the bytes, and without this
    # they share a key -- read_rows keeps the best of the two and the slower one vanishes,
    # which is exactly the comparison the single-precision arm exists to make.
    if r.get("bsr_storage", "n/a") not in ("n/a", "", None):
        parts.append("bsr " + r["bsr_storage"])
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


def spread_cell(r):
    s, n = r.get("_spread"), r.get("_n", 1)
    if s is None:
        return "1 run"
    return "%.0f%% of %d" % (s, n)


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
    best, seen, skipped = {}, {}, 0
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
                seen.setdefault(key, []).append(rate)
                if key not in best or rate > float(best[key]["MDOF_s"]):
                    best[key] = raw
    # How far apart the readings of one configuration were, as a percentage of the best.
    # Reporting only the best hides whether a 3% difference between two rows means anything:
    # the Rhie-Chow configurations here spread 20-32% across passes, which is more than the
    # gaps a reader would otherwise take for results.
    for key, row in best.items():
        rates = seen[key]
        row["_n"] = len(rates)
        row["_spread"] = (max(rates) - min(rates)) / max(rates) * 100 if len(rates) > 1 else None
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
    # Both layouts, not packed alone. Hardcoding `packed` here made the analysis code assert
    # the answer the campaign is meant to measure, and it is the standing convention in this
    # repository that a kernel number is reported for the standard and the packed layout side
    # by side. A layout with no rows simply produces no section.
    layouts = [l for l in ("packed", "atomic", "colored")
               if any((r.get("ran_layout") or r["layout"]) == l for r in rows)]
    for op, lay in [(o, l) for o in ("residual", "jac_action") for l in layouts]:
        # One variant, not one layout. A cascade is a table in which only the OPERATOR
        # varies, so mixing kernels or geometries into it puts two effects in one column --
        # which it did: the bare `element kernel` row appeared eight times, once per member
        # of the variant sweep, and none of them was a stage of anything.
        pool = [r for r in rows
                if r["operation"] == op and (r.get("ran_layout") or r["layout"]) == lay
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
                        spread_cell(r), "%.2fx" % (ref / rate) if rate > 0 else "–"])
        # The other way to apply the Jacobian belongs in this table, because the question
        # the table answers -- what does a matvec cost -- has two answers and one of them
        # is not matrix-free. One row per size and no more: the SpMV's cost is set by the
        # sparsity pattern, which is the mesh's node-to-node graph whatever terms the
        # values carry. The measurement below is the same reading for every set of terms,
        # which is what says so.
        # No layout filter: the SpMV has none. It reads a matrix, and --layout selects how
        # the matrix-free sweep visits elements, which is not a question a matvec over CSR
        # rows can answer differently. Filtering on one layout only dropped readings.
        # The storage type IS a distinction, so the two are reported separately.
        spmv = [r for r in rows if r["operation"] == "bsr_apply" and r["dofs"] == size]
        if op == "jac_action" and spmv:
            for storage in ("f64", "f32"):
                arm = [r for r in spmv if r.get("bsr_storage", "f64") == storage]
                if not arm:
                    continue
                rate = max(float(r["MDOF_s"]) for r in arm)
                spread = ((rate - min(float(r["MDOF_s"]) for r in arm)) / rate * 100) \
                    if len(arm) > 1 else None
                note = "SpMV of the assembled BSR, %s values" % storage
                if spread is not None:
                    note += " — %d term sets, %.1f%% apart" % (len(arm), spread)
                out.append(["assembled matrix", note, "warm", "%.0f" % rate,
                            "n=%d" % len(arm), "%.2fx" % (ref / rate) if rate > 0 else "–"])
        body.append("### %s, %s layout, %s dof\n\n" % (op, lay, "{:,}".format(size)) +
                    table(["operator", "terms carried", "working set", "MDOF/s",
                           "spread", "cost vs bare kernel"], out))
    if not body:
        return ""
    return ("## What the physics costs\n\n"
            "The element kernel alone against the same kernel carrying what the solver\n"
            "needs, one operation and one size at a time so that only the operator varies.\n"
            "The bare-kernel row is the number the regression gate tracks and the one every\n"
            "quoted figure has historically meant; the solver does not run it.\n\n"
            "`spread` is how far apart that configuration's repeated measurements were, as a\n"
            "percentage of the best. Two rows differ meaningfully only when the gap between\n"
            "them is larger than that -- which is not true of every pair here, and saying so\n"
            "is cheaper than inviting the reader to over-read a 3% difference.\n\n"
            + "\n".join(body))


def section_layout_matrix(rows):
    """The standard layout against the packed one at matched terms, with the SpMV beside them.

    The cascade above varies the operator and holds the layout still. This holds the operator
    still and varies the layout, which is the other question and the one the campaign was run
    to answer. Rows are paired on everything except the layout -- operation, size, kernel,
    terms and measurement condition -- so a pair is two readings of one operator and the
    ratio between them means what it appears to mean. An unpaired configuration is omitted
    rather than compared against the nearest thing available.

    The assembled matvec sits in the same row because it is the third way to apply the same
    Jacobian, and its two storage precisions sit beside each other because halving the value
    traffic is the only lever a bandwidth-bound SpMV has.
    """
    def pair_key(r):
        return (r["operation"], r["dofs"], r.get("ran_kernel", "?"),
                tuple(f(r) for _, _, f in TERMS), condition(r))

    fam = {}
    for r in rows:
        if r["operation"] == "bsr_apply":
            continue
        fam.setdefault(pair_key(r), {})[r.get("ran_layout") or r["layout"]] = r

    # Keyed on size and storage only: the SpMV is layout-free and carries whatever terms the
    # assembly put in the values, which does not change what a matvec over the pattern costs.
    spmv = {}
    for r in rows:
        if r["operation"] != "bsr_apply":
            continue
        k = (r["dofs"], r.get("bsr_storage", "f64"))
        if k not in spmv or float(r["MDOF_s"]) > float(spmv[k]["MDOF_s"]):
            spmv[k] = r

    def rate(r):
        return float(r["MDOF_s"]) if r is not None else None

    out = []
    for key in sorted(fam, key=lambda k: (k[0], k[1])):
        group = fam[key]
        if "packed" not in group:
            continue
        p_r = rate(group["packed"])
        for lay in ("atomic", "colored"):
            if lay not in group:
                continue
            o_r = rate(group[lay])
            f64, f32 = rate(spmv.get((key[1], "f64"))), rate(spmv.get((key[1], "f32")))
            # The terms, not only the completeness word. Two operators that are both
            # "partial" -- Rhie-Chow alone, and Rhie-Chow with the pressure gradient rebuilt
            # per apply -- are different operators with different costs, and with only the
            # word they arrive here as two rows that look identical and disagree.
            terms = ", ".join("%s (%s)" % (h, f(group["packed"])) if f(group["packed"]) != "carried" else h
                              for _, h, f in TERMS if f(group["packed"]) not in ("–", "?"))
            out.append(["{:,}".format(key[1]), key[0], key[2],
                        completeness(group["packed"]), terms or "none",
                        "%.0f" % p_r, lay, "%.0f" % o_r,
                        "%.2fx" % (p_r / o_r) if o_r else "–",
                        "%.0f" % f64 if f64 else "–",
                        "%.0f" % f32 if f32 else "–",
                        "%.2fx" % (f32 / f64) if f64 and f32 else "–",
                        spread_cell(group["packed"])])
    if not out:
        return ""
    return ("## Standard against packed\n\n"
            "One operator per row, measured on both layouts. The ratio is packed over the\n"
            "standard layout, so above 1.00x the packed sweep is the faster of the two.\n\n"
            "`SpMV f64` and `SpMV f32` are the assembled matrix applied at the same size --\n"
            "the same Jacobian, neither layout, and the only one of the three whose cost is\n"
            "set by the sparsity pattern rather than by the element kernel. `f32 vs f64` is\n"
            "what halving the value traffic bought; the arithmetic is identical, since the\n"
            "SpMV up-converts each entry and accumulates in double either way.\n\n"
            "The final column is the spread of the packed readings. A ratio nearer to 1.00x\n"
            "than that spread is a tie and must be read as one.\n\n"
            + table(["dof", "operation", "kernel", "operator", "terms carried", "packed MDOF/s",
                     "standard", "standard MDOF/s", "packed vs standard",
                     "SpMV f64", "SpMV f32", "f32 vs f64", "spread (packed)"], out))


def section_provenance(rows, paths, skipped, prose=()):
    hosts = sorted({r.get("host", "?") for r in rows})
    facts = [["generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
             ["rows", str(len(rows))],
             ["machines", ", ".join(hosts) if hosts else "?"],
             ["sources", ", ".join(os.path.basename(p) for p in paths)]]
    if skipped:
        facts.append(["rows skipped (no ran_* columns)", str(skipped)])
    # The regenerate line MUST carry the --prose arguments this run was given. Without them it
    # is an instruction to DESTROY the hand-written sections: the tables are rebuilt from
    # scratch every time and anything not passed through --prose does not come back. A reader
    # following the report's own instructions would have silently deleted five sections, which
    # is the most expensive kind of wrong a provenance note can be.
    cmd = "python3 python/cvfem_kernel_report.py %s -o docs/CVFEM_Kernels.md --html" % " ".join(paths)
    for f in prose:
        cmd += " \\\n      --prose %s" % f
    if prose:
        facts.append(["hand-written sections", "%d, carried through --prose" % len(prose)])
    return ("## Provenance\n\n" + table(["field", "value"], facts) +
            "\nRegenerate with:\n\n```\n" + cmd + "\n```\n")


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
    for s in (section_coverage(rows), section_throughput(rows), section_cascade(rows),
              section_layout_matrix(rows)):
        if s.strip():
            doc.append(s)
            doc.append("")
    for p in prose:
        with open(p) as fh:
            doc.append(fh.read().rstrip("\n"))
        doc.append("")
    doc.append(section_provenance(rows, paths, skipped, prose))
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
             "MDOF_s": "100", "dofs": "1000", "threads": "72", "host": "h",
             "bsr_storage": "n/a"}
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
    check(variant(row(bsr_storage="n/a")) == variant(row()),
          "a run that built no matrix says nothing about storage")
    # --layout store falls through to the packed sweep for residual and jac_action, so a row
    # that asked for store and ran packed must key as packed -- otherwise one call site sits
    # in the table twice and the noise between the two readings looks like a result.
    check(variant(row(layout="store", ran_layout="packed")) == variant(row(layout="packed")),
          "a layout that fell back is keyed by what ran, not what was asked")
    check(variant(row(layout="store", ran_layout="store")) != variant(row(layout="packed")),
          "a layout that did run on its own stays distinct")
    check(variant(row(operation="bsr_apply", bsr_storage="f32")) !=
          variant(row(operation="bsr_apply", bsr_storage="f64")),
          "the two SpMV precisions are different code paths")

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

        # The failure this guards: two SpMV runs of one matrix at two storage precisions
        # sharing a key, read_rows keeping only the faster, and the comparison the
        # single-precision arm exists to make disappearing without a diagnostic.
        spmv = os.path.join(d, "spmv.csv")
        with open(spmv, "w") as fh:
            fh.write(hdr + ",bsr_storage\n")
            fh.write("bsr_apply,packed,sumfact,affine,1000,50,72,h,n/a,off,off,0,0,0,f64\n")
            fh.write("bsr_apply,packed,sumfact,affine,1000,80,72,h,n/a,off,off,0,0,0,f32\n")
        rows, _ = read_rows([spmv])
        check(len(rows) == 2, "the two SpMV precisions survive as two rows")

        # And the layout matrix: a packed row with no standard partner must be omitted, not
        # paired against whatever else is in the file.
        lay = os.path.join(d, "layout.csv")
        with open(lay, "w") as fh:
            fh.write(hdr + ",bsr_storage\n")
            fh.write("jac_action,packed,sumfact,affine,1000,200,72,h,sumfact,off,off,0,0,0,n/a\n")
            fh.write("jac_action,atomic,sumfact,affine,1000,100,72,h,sumfact,off,off,0,0,0,n/a\n")
            fh.write("jac_action,packed,sumfact,affine,2000,300,72,h,sumfact,on,off,0,0,0,n/a\n")
        rows, _ = read_rows([lay])
        m = section_layout_matrix(rows)
        check("2.00x" in m, "a matched pair reports the ratio between the layouts")
        check(m.count("|\n") - 2 == 1, "the unpaired packed row is omitted, not paired")

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
