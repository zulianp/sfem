"""The article's figures, from the Grace logs in measured/.

    python paper_figures.py measured/ ../../../docs/compressed_partial_assembly/figures \
        measured_qsym/ measured_qbsr/

The further directories hold the quantized-BSR campaigns (results_tables.py).

Vector PDFs at the article's text width, in its serif fonts.  Every value is parsed from the
logs (results_tables.py); nothing is typed in.  Operators whose logs are not yet present are
left out of a panel rather than drawn as zero.
"""

import glob
import os
import re
import sys

import matplotlib
import matplotlib.patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shutil
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_tables import (ELEMENTS, NAMES, bench_rows, newton_case, operator_name,  # noqa: E402
                           parse_bench_sweep, parse_newton, parse_newton_sweep)

TEXTWIDTH = 6.27  # inches, a4paper with 1in margins
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 6.8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1, "lines.markersize": 3.5, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "legend.frameon": False,
})
# The article's own fonts, through LaTeX, when it is installed.
if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.serif": ["Computer Modern Roman"],
                         "text.latex.preamble": r"\usepackage{amsmath}"})


def plain_log_axis(ax, axis="y"):
    """Log scale labelled 0.5, 1, 2, ... rather than 5 x 10^-1."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    a.set_major_formatter(fmt)
    a.set_minor_formatter(NullFormatter())


def dof_axis(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"))
    ax.xaxis.set_minor_formatter(NullFormatter())

# One colour per operator family, kept across every figure.
STYLE = {
    "BSR f64": dict(label="BSR, fp64", color="#252525", hatch=None, marker="o", ls="-"),
    "exact": dict(label="exact (BSR = matrix-free = fp64 store)", color="#252525", hatch=None, marker="o", ls="-"),
    "BSR f32": dict(label="BSR, fp32", color="#8c8c8c", hatch=None, marker="o", ls="--"),
    "MF": dict(label="matrix-free (exact)", color="#b15928", hatch="xx", marker="h", ls="-."),
    "P0": dict(label="projected (approx.)", color="#d95f02", hatch="///", marker="s", ls=":"),
    "qp f16": dict(label="points/moments, fp16", color="#1b9e77", hatch=None, marker="^", ls="-"),
    "qp f16md": dict(label="points/moments, fp16 mean+dev.", color="#66a61e", hatch=None, marker="v", ls="-"),
    "qp i8md": dict(label="points/moments, int8 mean+dev.", color="#7570b3", hatch=None, marker="d", ls="-"),
    "qp i8": dict(label="points/moments, int8 per element", color="#e7298a", hatch="..", marker="x", ls=":"),
    "ea diff f16": dict(label="element matrix, fp16 diff.", color="#1f78b4", hatch=None, marker="P", ls="-"),
    "ea diff i8": dict(label="element matrix, int8 diff.", color="#a6cee3", hatch=None, marker="X", ls="-"),
    "ea full f16": dict(label="element matrix, fp16 full", color="#6a3d9a", hatch="\\\\\\", marker="*", ls="--"),
    "ea pair i8": dict(label="element matrix, int8 diff., pair scale", color="#08306b", hatch=None, marker="D",
                       ls="-"),
    "qbsr f16": dict(label="quantized BSR, fp16", color="#e6ab02", hatch=None, marker="<", ls="-"),
    "qbsr i8": dict(label="quantized BSR, int8", color="#a6761d", hatch=None, marker=">", ls="-"),
}
# The bench rows behind each operator (packed scatter for the element operators).
BENCH_KEY = {
    "MF": "exact matrix-free packed two-pass", "BSR f64": "BSR f64", "BSR f32": "BSR f32", "P0": "projected P0 f32 packed",
    "qp f16": "qp f16 blk a32 lanes packed", "qp f16md": "qp f16 mean+dev blk a32 lanes packed",
    "qp i8md": "qp i8 mean+dev blk a32 lanes packed", "qp i8": "qp i8/el blk a32 lanes packed",
    "ea diff f16": "ea diff f16 packed", "ea diff i8": "ea diff i8 packed", "ea full f16": "ea full f16 packed",
    "ea pair i8": "ea diff i8 pair packed", "qbsr f16": "BSR f16 upper u16", "qbsr i8": "BSR i8/blk upper u16",
}
# The Newton labels behind each operator.
NEWTON_KEY = {
    "MF": "matrix-free", "exact": "BSR f64", "BSR f64": "BSR f64", "BSR f32": "BSR f32", "P0": "P0 projected", "qp f16": "qp f16",
    "qp f16md": "qp f16md", "qp i8md": "qp i8md", "qp i8": "qp i8", "ea diff f16": "qp ead16",
    "ea diff i8": "qp eadi8", "ea full f16": "qp eaf16", "ea pair i8": "qp eadi8p", "qbsr f16": "qp bsro16",
    "qbsr i8": "qp bsroi8",
}
# Nodes per element: the pair-scaled store's side data is one byte per node pair.
NODES = {"TET4": 4, "TET10": 10, "HEX8": 8, "PROTEUS_HEX27": 27}


def display(el):
    return NAMES.get(el, el)


def bars(ax, ops, value, ylabel, log=False, annotate="{:.0f}", elements=ELEMENTS, text=None):
    """Grouped bars: one group per element, one bar per operator that has a value anywhere.
    `text(o, el)`, when given, replaces the formatted value above each bar."""
    present = [o for o in ops if any(value(o, el) is not None for el in elements)]
    x = np.arange(len(elements))
    w = 0.84 / max(1, len(present))
    top = 0
    for k, o in enumerate(present):
        st = STYLE[o]
        vals = [value(o, el) for el in elements]
        pos = x - 0.42 + w * (k + 0.5)
        hv = [v if v is not None else np.nan for v in vals]
        ax.bar(pos, hv, w * 0.92, color=st["color"], hatch=st["hatch"], edgecolor="white" if not st["hatch"] else
               "#ffffff", linewidth=0.3, label=st["label"])
        for p, v, el in zip(pos, vals, elements):
            if v is None:
                continue
            top = max(top, v)
            if text:
                ax.text(p, v, " " + text(o, el), ha="center", va="bottom", rotation=90, fontsize=5.2)
            elif annotate:
                ax.text(p, v, " " + annotate.format(v), ha="center", va="bottom", rotation=90, fontsize=5.6)
    ax.set_xticks(x, [display(e) for e in elements])
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, top * (1.45 if text else 1.22) if top else 1)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    return present


def legend_below(fig, axes, ncol=4):
    handles, labels = [], []
    for ax in np.atleast_1d(axes).flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.02))


MEMORY_OPS = [
    # label, bench row, kind, code bytes, extras
    ("BSR, fp64", "BSR f64", "bsr", 8, None),
    ("BSR, fp32", "BSR f32", "bsr", 4, None),
    ("points/moments, fp16", "qp f16 blk a32 lanes packed", "store", 2, "scale"),
    ("points/moments, fp16 mean+dev.", "qp f16 mean+dev blk a32 lanes packed", "store", 2, "mean"),
    ("points/moments, int8 mean+dev.", "qp i8 mean+dev blk a32 lanes packed", "store", 1, "mean"),
    ("element matrix, fp16 full", "ea full f16 packed", "store", 2, "scale"),
    ("element matrix, fp16 diff.", "ea diff f16 packed", "store", 2, "scale"),
    ("element matrix, int8 diff.", "ea diff i8 packed", "store", 1, "scale"),
    ("element matrix, int8 diff., pair scale", "ea diff i8 pair packed", "store", 1, "pair"),
    ("quantized BSR, fp16", "BSR f16 upper u16", "qbsr", 2, None),
    ("quantized BSR, int8", "BSR i8/blk upper u16", "qbsr", 1, None),
]
PART_COLOURS = {"values": "#4d4d4d", "indices": "#b3b3b3", "codes": "#1b9e77", "scales": "#e6ab02",
                "mean": "#7570b3"}


def memory_figure(rows, meta, out, src=None):
    # Bytes per dof do not depend on the problem size: a row missing at the benchmark size may come
    # from a smaller run (membench_*), for this figure only.
    if src:
        for path in sorted(glob.glob(os.path.join(src, "membench_*.out"))):
            for el, dof, rr in parse_bench_sweep(path):
                for k, v in rr.items():
                    rows.setdefault(el, {}).setdefault(k, v)
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 5.0), sharey=True)
    for ax, el in zip(axes.flat, ELEMENTS):
        r = rows.get(el, {})
        m = meta.get(el)
        if not m:
            continue
        el_per_dof = m["elements"] / m["dof"]
        labels, parts = [], []
        for label, key, kind, cb, extra in MEMORY_OPS:
            labels.append(label)
            if key not in r:
                parts.append(None)
                continue
            total = r[key]["bytes_per_dof"]
            if kind == "bsr":
                # per block: 9 values and a 4-byte column index; row pointers are the remainder
                vals = total * 9 * cb / (9 * cb + 4)
                p = {"values": vals, "indices": total - vals}
            elif kind == "qbsr":
                # per stored block: 9 codes, a 4-byte scale (int8) and a 2-byte column offset; the
                # remainder's indices, the halo and the row order are counted with the offsets
                sc = 4 if cb == 1 else 0
                codes = total * 9 * cb / (9 * cb + sc + 2)
                scales = total * sc / (9 * cb + sc + 2)
                p = {"codes": codes, "scales": scales, "indices": total - codes - scales}
            else:
                pairs = NODES[el] * (NODES[el] - 1) / 2
                side = {"scale": 4.0, "mean": 4.0 + 45 * 4.0, "pair": 4.0 + pairs}[extra] * el_per_dof
                p = {"codes": total - side}
                if extra == "mean":
                    p["mean"] = 45 * 4.0 * el_per_dof
                    p["scales"] = 4.0 * el_per_dof
                else:
                    p["scales"] = side
            parts.append((total, p))
        y = np.arange(len(labels))[::-1]
        for yi, part in zip(y, parts):
            if part is None:
                ax.text(0, yi, " n/a", va="center", fontsize=6, color="#808080")
                continue
            total, p = part
            left = 0
            for name in ("values", "indices", "codes", "mean", "scales"):
                if name in p and p[name] > 0:
                    ax.barh(yi, p[name], left=left, color=PART_COLOURS[name], height=0.7, linewidth=0)
                    left += p[name]
            ax.text(left, yi, f" {total:.0f}", va="center", fontsize=6)
        ax.set_yticks(y, labels)
        ax.set_title(display(el))
        ax.set_xlabel("bytes per dof")
        xmax = max(pt[0] for pt in parts if pt) * 1.25
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
        ax.set_axisbelow(True)
        # absolute size at the benchmark's problem size
        sec = ax.secondary_xaxis("top", functions=(lambda b, d=m["dof"]: b * d / 1e9,
                                                   lambda g, d=m["dof"]: g * 1e9 / d))
        sec.set_xlabel(f"GB at {m['dof'] / 1e6:.1f}M dof", fontsize=7)
        sec.tick_params(labelsize=6.5)
    handles = [matplotlib.patches.Patch(color=c, label=n) for n, c in
               (("BSR values", PART_COLOURS["values"]), ("column indices, offsets and row pointers", PART_COLOURS["indices"]),
                ("encoded numbers", PART_COLOURS["codes"]), ("f32 element mean", PART_COLOURS["mean"]),
                ("scales", PART_COLOURS["scales"]))]
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(os.path.join(out, "memory_footprint.pdf"))
    plt.close(fig)


def main():
    src, out, extras = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out, exist_ok=True)
    rows, meta = bench_rows(src, extras)

    def bench(field):
        return lambda o, el: rows.get(el, {}).get(BENCH_KEY[o], {}).get(field)

    apply_ops = ["BSR f64", "BSR f32", "MF", "P0", "qp f16", "qp i8md", "ea diff f16", "ea diff i8", "ea pair i8",
                 "qbsr f16", "qbsr i8"]

    # Figure: throughput and memory at saturation.
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.6))
    bars(ax, apply_ops, bench("mdofs"), "MDOF/s (higher is better)")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    legend_below(fig, [ax])
    fig.savefig(os.path.join(out, "throughput.pdf"))
    plt.close(fig)

    # Figure: operator memory, decomposed.  Totals are the measured bytes; the split follows the
    # formats (BSR: 9 values and one 4-byte column index per block, plus row pointers; stores:
    # codes, one 4-byte scale per element or per component, and a 45-number f32 mean).
    memory_figure({el: dict(r) for el, r in rows.items()}, meta, out, src)

    # Figure: set-up per Newton step, and the operator error.
    setup_key = {"BSR f64": "BSR f64", "BSR f32": "BSR f32", "P0": "projected P0 f32",
                 "qp f16": "qp f16 blk a32 lanes", "qp i8md": "qp i8 mean+dev blk a32 lanes",
                 "ea diff f16": "ea diff f16 atomic", "qbsr i8": "BSR i8/blk upper u16"}
    err_ops = ["P0", "qp f16", "qp f16md", "qp i8md", "qp i8", "ea full f16", "ea diff f16", "ea diff i8",
               "ea pair i8", "qbsr f16", "qbsr i8"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7))
    bars(axes[0], list(setup_key), lambda o, el: rows.get(el, {}).get(setup_key[o], {}).get("setup_ms"),
         "ms per Newton step")
    axes[0].set_title("Set-up (lower is better)")
    bars(axes[1], err_ops, bench("energy"), "relative energy error", log=True, annotate=None)
    axes[1].yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[1].set_title("Operator error, mild state, white-noise increment")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    legend_below(fig, axes)
    fig.savefig(os.path.join(out, "setup_error.pdf"))
    plt.close(fig)

    # Figure: Newton and CG iterations at about 1M dof (stretch case) and the twist case.
    def newton_logs(case):
        # A long case may be split over several logs; the quantized BSR runs come from the extras.
        return {el: runs for el, (runs, m) in newton_case(src, extras, case).items()}

    for case, name, title in (("1M", "newton_iterations.pdf", "stretch and shear"),
                              ("twist", "newton_twist.pdf", "90$^\\circ$ twist")):
        logs = newton_logs(case)
        if not logs:
            continue
        ops = ["exact", "P0", "qp f16", "qp f16md", "qp i8md", "qp i8", "ea full f16", "ea diff f16", "ea diff i8",
               "ea pair i8", "qbsr f16", "qbsr i8"]

        def it(field):
            return lambda o, el: (logs.get(el, {}).get(NEWTON_KEY[o]) or {}).get(field)

        els = [el for el in ELEMENTS if el in logs]
        fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))
        bars(axes[0], ops, it("newton"), "Newton iterations", elements=els)
        axes[0].set_title(f"Newton iterations, {title}")
        bars(axes[1], ops, it("linear"), "CG iterations", annotate=None, elements=els)
        axes[1].set_title(f"CG iterations, {title}")
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        legend_below(fig, axes)
        fig.savefig(os.path.join(out, name))
        plt.close(fig)

    # Figure: time to solution relative to BSR f32, for every case with timings: the stretch at about
    # 1M and 10M dof and the twist at about 1M.  Every run is shown; each bar carries its Newton
    # count, so a run whose load-step path differs from the reference's is visible as such.
    cases = [("1M", "stretch and shear, about $10^6$ dof"),
             ("twist", "90$^\\circ$ twist, about $10^6$ dof"),
             ("10M", "stretch and shear, about $10^7$ dof")]
    cases = [(newton_logs(p), t) for p, t in cases]
    cases = [(l, t) for l, t in cases if l]
    if cases:
        ops = ["BSR f64", "BSR f32", "MF", "P0", "qp f16", "qp f16md", "qp i8md", "ea full f16", "ea diff f16",
               "ea diff i8", "ea pair i8", "qbsr f16", "qbsr i8"]
        fig, axes = plt.subplots(len(cases), 1, figsize=(TEXTWIDTH, 2.25 * len(cases)), squeeze=False)
        for ax, (logs, title) in zip(axes[:, 0], cases):
            def rel(o, el, logs=logs):
                runs = logs.get(el, {})
                r, ref = runs.get(NEWTON_KEY[o]), runs.get("BSR f32")
                return r["seconds"] / ref["seconds"] if r and ref else None

            def label(o, el, logs=logs):
                return f"{rel(o, el):.2f} ({logs[el][NEWTON_KEY[o]]['newton']})"

            els = [el for el in ELEMENTS if el in logs]
            bars(ax, ops, rel, "time / BSR fp32", elements=els, text=label, log=True)
            plain_log_axis(ax)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 3)
            ax.axhline(1, color="#8c8c8c", lw=0.6)
            ax.set_title(f"Newton time to solution, {title}")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        legend_below(fig, axes, ncol=4)
        fig.savefig(os.path.join(out, "time_to_solution.pdf"))
        plt.close(fig)

    # Figures: resolution sweeps.
    sweep = {}
    auto = sorted(glob.glob(os.path.join(src, "bsweepauto_*.out")))
    for path in sorted(glob.glob(os.path.join(src, "bsweep_*.out"))) + auto:
        for el, dof, r in parse_bench_sweep(path):
            sweep.setdefault(el, {}).setdefault(dof, {}).update(r)
    if sweep:
        fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 4.4), sharey=True)
        for ax, el in zip(axes.flat, ELEMENTS):
            pts = sorted(sweep.get(el, {}).items())
            for o in ["BSR f64", "BSR f32", "MF", "P0", "qp f16", "qp i8md", "ea diff f16"]:
                xs = [d for d, r in pts if BENCH_KEY[o] in r]
                ys = [r[BENCH_KEY[o]]["mdofs"] for d, r in pts if BENCH_KEY[o] in r]
                if xs:
                    st = STYLE[o]
                    ax.plot(xs, ys, marker=st["marker"], ls=st["ls"], color=st["color"], label=st["label"])
            ax.set_xscale("log")
            dof_axis(ax)
            ax.set_title(display(el))
            ax.set_xlabel("dof")
            ax.grid(lw=0.3, alpha=0.5)
        for a in axes[:, 0]:
            a.set_ylabel("application, MDOF/s")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        legend_below(fig, axes, ncol=4)
        fig.savefig(os.path.join(out, "sweep_throughput.pdf"))
        plt.close(fig)

    nsweep = {}
    auto = sorted(glob.glob(os.path.join(src, "nsweepauto_*.out")))
    for path in sorted(glob.glob(os.path.join(src, "nsweep_*.out"))) + auto:
        el = re.search(r"nsweep(?:auto)?_(\w+?)_", os.path.basename(path)).group(1).upper()
        el = {"HEX27": "PROTEUS_HEX27"}.get(el, el)
        for dof, runs in parse_newton_sweep(path):
            nsweep.setdefault(el, {})[dof] = runs
    # The saturated runs extend the sweep, unless the reference operators themselves took different
    # Newton paths (step cuts that depend on rounding), which makes the run a comparison of paths.
    for path in sorted(glob.glob(os.path.join(src, "newton_*_10M_*.out"))):
        runs, m = parse_newton(path)
        runs = {operator_name(l): r for l, r in runs.items() if r["seconds"] is not None}
        nsweep.setdefault(m["element"], {}).setdefault(m["dof"], {}).update(runs)
    for el in nsweep:
        nsweep[el] = {d: r for d, r in nsweep[el].items() if not (
            r.get("BSR f64") and r.get("BSR f32") and r["BSR f64"]["newton"] != r["BSR f32"]["newton"])}
    if nsweep:
        fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 4.4), sharey=True)
        for ax, el in zip(axes.flat, ELEMENTS):
            pts = sorted(nsweep.get(el, {}).items())
            for o in ["BSR f64", "P0", "qp f16", "qp f16md", "qp i8md", "ea diff f16", "ea pair i8"]:
                xs, ys = [], []
                for dof, runs in pts:
                    r, ref = runs.get(NEWTON_KEY[o]), runs.get("BSR f32")
                    if r and ref and r["seconds"] and ref["seconds"]:
                        xs.append(dof)
                        ys.append(r["seconds"] / ref["seconds"])
                if xs:
                    st = STYLE[o]
                    ax.plot(xs, ys, marker=st["marker"], ls=st["ls"], color=st["color"], label=st["label"])
            ax.axhline(1, color="#8c8c8c", lw=0.6)
            ax.set_xscale("log")
            ax.set_yscale("log")
            dof_axis(ax)
            plain_log_axis(ax)
            ax.set_title(display(el))
            ax.set_xlabel("dof")
            ax.grid(lw=0.3, alpha=0.5, which="both")
        for a in axes[:, 0]:
            a.set_ylabel("time to solution / BSR fp32")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        legend_below(fig, axes, ncol=4)
        fig.savefig(os.path.join(out, "sweep_newton.pdf"))
        plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main()
