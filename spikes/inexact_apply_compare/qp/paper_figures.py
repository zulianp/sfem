"""The article's figures, from the Grace logs in measured/ and the H100 logs in measured_gpu/.

    python paper_figures.py measured/ ../../../docs/compressed_partial_assembly/figures \
        measured_qsym/ measured_qbsr/
    python paper_figures.py --machine gpu measured_gpu/ ../../../docs/compressed_partial_assembly/figures

The GPU's figures are the same figures, the files named *_gpu.pdf.

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
from results_tables import (ELEMENTS, GPU_BENCH, NAMES, REFERENCE_BSR, bench_rows, newton_case,  # noqa: E402
                           operator_name, parse_bench_sweep, parse_newton, parse_newton_sweep)

# "cpu" (Grace) or "gpu" (H100): which campaign the figures are drawn from.
MACHINE = "cpu"


def figure_name(name):
    return name if MACHINE == "cpu" else name.replace(".pdf", "_gpu.pdf")

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

# The formats every figure shows, in this order: the references, then the compressed formats from the
# assembled end to the quadrature points.  One colour family per format, darker for fp16.
FORMATS = ["BSR f64", "BSR f32", "qbsr f16", "qbsr i8", "ea diff f16", "ea pair i8", "qp f16", "qp i8md"]
STYLE = {
    "BSR f64": dict(label="BSR, fp64", color="#252525", hatch=None, marker="o", ls="-"),
    "BSR f32": dict(label="BSR, fp32", color="#8c8c8c", hatch=None, marker="o", ls="--"),
    "qbsr f16": dict(label="quantized BSR, fp16", color="#a6761d", hatch=None, marker="<", ls="-"),
    "qbsr i8": dict(label="quantized BSR, int8", color="#e6ab02", hatch=None, marker=">", ls="--"),
    "ea diff f16": dict(label="element matrix, fp16", color="#1f78b4", hatch=None, marker="P", ls="-"),
    "ea pair i8": dict(label="element matrix, int8 (pair scale)", color="#a6cee3", hatch=None, marker="D", ls="--"),
    "qp f16": dict(label="quadrature points, fp16", color="#1b9e77", hatch=None, marker="^", ls="-"),
    "qp i8md": dict(label="quadrature points, int8 mean+dev.", color="#99d8a8", hatch=None, marker="d", ls="--"),
}
# The bench rows behind each format (packed scatter for the element formats).
BENCH_KEY = {
    "BSR f64": "BSR f64", "BSR f32": "BSR f32", "qbsr f16": "BSR f16 upper u16", "qbsr i8": "BSR i8/blk upper u16",
    "ea diff f16": "ea diff f16 packed", "ea pair i8": "ea diff i8 pair packed",
    "qp f16": "qp f16 blk a32 lanes packed", "qp i8md": "qp i8 mean+dev blk a32 lanes packed",
}
# Their set-up rows (the element matrix's set-up is timed with the atomic scatter).
SETUP_KEY = {
    "BSR f64": "BSR f64", "BSR f32": "BSR f32", "qbsr f16": "BSR f16 upper u16", "qbsr i8": "BSR i8/blk upper u16",
    "ea diff f16": "ea diff f16 atomic", "qp f16": "qp f16 blk a32 lanes", "qp i8md": "qp i8 mean+dev blk a32 lanes",
}
# The Newton labels behind each format.
NEWTON_KEY = {
    "BSR f64": "BSR f64", "BSR f32": "BSR f32", "qbsr f16": "qp bsro16", "qbsr i8": "qp bsroi8",
    "ea diff f16": "qp ead16", "ea pair i8": "qp eadi8p", "qp f16": "qp f16", "qp i8md": "qp i8md",
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
    """One legend for all panels, in the order of FORMATS (one column per family)."""
    handles, labels = [], []
    for ax in np.atleast_1d(axes).flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    order = {STYLE[o]["label"]: k for k, o in enumerate(FORMATS)}
    pairs = sorted(zip(handles, labels), key=lambda hl: order.get(hl[1], len(order)))
    fig.legend([h for h, _ in pairs], [l for _, l in pairs], loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, -0.02))


# How each format's bytes split: kind, code bytes, side data.
MEMORY_KIND = {
    "BSR f64": ("bsr", 8, None), "BSR f32": ("bsr", 4, None), "qbsr f16": ("qbsr", 2, None),
    "qbsr i8": ("qbsr", 1, None), "ea diff f16": ("store", 2, "scale"), "ea pair i8": ("store", 1, "pair"),
    "qp f16": ("store", 2, "scale"), "qp i8md": ("store", 1, "mean"),
}
MEMORY_OPS = [(STYLE[o]["label"], BENCH_KEY[o]) + MEMORY_KIND[o] for o in FORMATS]
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
                # remainder's indices, the halo and the row order are counted with the offsets.  The
                # GPU's packed records: int8 16 bytes (a byte of padding), f16 20; on HEX27 a 4-byte
                # column besides.
                sc = 4 if cb == 1 else 0
                idx = 2 if MACHINE == "cpu" else (3 if cb == 1 else 2) + (4 if el == "PROTEUS_HEX27" else 0)
                codes = total * 9 * cb / (9 * cb + sc + idx)
                scales = total * sc / (9 * cb + sc + idx)
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
    fig.savefig(os.path.join(out, figure_name("memory_footprint.pdf")))
    plt.close(fig)


def file_gpu_rows(r):
    """The GPU's rows of one benchmark, also under the CPU's keys, so every figure reads them alike."""
    for o in FORMATS:
        for cand in GPU_BENCH[o]:
            if cand in r:
                r.setdefault(BENCH_KEY[o], r[cand])
                if o in SETUP_KEY:
                    r.setdefault(SETUP_KEY[o], r[cand])
                break
    return r


def main():
    global MACHINE
    args = sys.argv[1:]
    if args[0] == "--machine":
        MACHINE = args[1]
        args = args[2:]
    src, out, extras = args[0], args[1], args[2:]
    os.makedirs(out, exist_ok=True)
    rows, meta = bench_rows(src, extras)
    if MACHINE == "gpu":
        for r in rows.values():
            file_gpu_rows(r)

    def bench(field):
        return lambda o, el: rows.get(el, {}).get(BENCH_KEY[o], {}).get(field)

    # Figure: throughput and memory at saturation.
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.6))
    bars(ax, FORMATS, bench("mdofs"), "MDOF/s (higher is better)")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    legend_below(fig, [ax])
    fig.savefig(os.path.join(out, figure_name("throughput.pdf")))
    plt.close(fig)

    # Figure: operator memory, decomposed.  Totals are the measured bytes; the split follows the
    # formats (BSR: 9 values and one 4-byte column index per block, plus row pointers; stores:
    # codes, one 4-byte scale per element or per component, and a 45-number f32 mean).
    memory_figure({el: dict(r) for el, r in rows.items()}, meta, out, src)

    # Figure: set-up per Newton step, and the operator error.
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7))
    bars(axes[0], FORMATS, lambda o, el: rows.get(el, {}).get(SETUP_KEY.get(o), {}).get("setup_ms") or None,
         "ms per Newton step")
    axes[0].set_title("Set-up (lower is better)")
    bars(axes[1], FORMATS, bench("energy"), "relative energy error", log=True, annotate=None)
    axes[1].yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[1].set_title("Operator error, mild state, white-noise increment")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    legend_below(fig, axes)
    fig.savefig(os.path.join(out, figure_name("setup_error.pdf")))
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
        ops = FORMATS

        def it(field):
            return lambda o, el: (logs.get(el, {}).get(NEWTON_KEY[o]) or {}).get(field)

        els = [el for el in ELEMENTS if el in logs]
        fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))
        bars(axes[0], ops, it("newton"), "Newton iterations", elements=els, log=True)
        plain_log_axis(axes[0])
        axes[0].set_title(f"Newton iterations, {title}")
        bars(axes[1], ops, it("linear"), "CG iterations", annotate=None, elements=els, log=True)
        axes[1].set_title(f"CG iterations, {title}")
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        legend_below(fig, axes)
        fig.savefig(os.path.join(out, figure_name(name)))
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
        ops = FORMATS
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
        fig.savefig(os.path.join(out, figure_name("time_to_solution.pdf")))
        plt.close(fig)

    # Figures: resolution sweeps.
    sweep = {}
    auto = sorted(glob.glob(os.path.join(src, "bsweepauto_*.out")))
    for path in sorted(glob.glob(os.path.join(src, "bsweep_*.out"))) + auto:
        for el, dof, r in parse_bench_sweep(path):
            sweep.setdefault(el, {}).setdefault(dof, {}).update(file_gpu_rows(r) if MACHINE == "gpu" else r)
    # The quantized BSR rows come from their own sweep in the extras, at the same sizes.
    for d in extras:
        for path in sorted(glob.glob(os.path.join(d, "ubsweep_*.out"))):
            for el, dof, r in parse_bench_sweep(path):
                for k, v in r.items():
                    if k.startswith("BSR ") and k not in REFERENCE_BSR:
                        sweep.setdefault(el, {}).setdefault(dof, {}).setdefault(k, v)
    if sweep:
        fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 4.4), sharey=True)
        for ax, el in zip(axes.flat, ELEMENTS):
            pts = sorted(sweep.get(el, {}).items())
            for o in FORMATS:
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
        fig.savefig(os.path.join(out, figure_name("sweep_throughput.pdf")))
        plt.close(fig)

    nsweep = {}
    auto = sorted(glob.glob(os.path.join(src, "nsweepauto_*.out")))
    for path in sorted(glob.glob(os.path.join(src, "nsweep_*.out"))) + auto:
        el = re.search(r"nsweep(?:auto)?_(\w+?)_", os.path.basename(path)).group(1).upper()
        el = {"HEX27": "PROTEUS_HEX27"}.get(el, el)
        for dof, runs in parse_newton_sweep(path):
            nsweep.setdefault(el, {})[dof] = runs
    # The quantized BSR runs come from their own sweep in the extras, at the same sizes.
    for d in extras:
        for path in sorted(glob.glob(os.path.join(d, "unsweep_*.out"))):
            el = re.search(r"unsweep_(\w+?)_", os.path.basename(path)).group(1).upper()
            el = {"HEX27": "PROTEUS_HEX27"}.get(el, el)
            for dof, runs in parse_newton_sweep(path):
                for op, r in runs.items():
                    if op.startswith("qp bsr") and r["seconds"] is not None:
                        nsweep.setdefault(el, {}).setdefault(dof, {}).setdefault(op, r)
    # The saturated runs extend the sweep, unless the reference operators themselves took different
    # Newton paths (step cuts that depend on rounding), which makes the run a comparison of paths.
    for el, (runs, m) in newton_case(src, extras, "10M").items():
        nsweep.setdefault(el, {}).setdefault(m["dof"], {}).update(runs)
    for el in nsweep:
        nsweep[el] = {d: r for d, r in nsweep[el].items() if not (
            r.get("BSR f64") and r.get("BSR f32") and r["BSR f64"]["newton"] != r["BSR f32"]["newton"])}
    if nsweep:
        fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 4.4), sharey=True)
        for ax, el in zip(axes.flat, ELEMENTS):
            pts = sorted(nsweep.get(el, {}).items())
            for o in FORMATS:
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
        fig.savefig(os.path.join(out, figure_name("sweep_newton.pdf")))
        plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main()
