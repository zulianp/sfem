"""Charts of the Grace measurements in measured/.

    python plot_results.py measured/ OUT_DIR

Throughput, operator memory and set-up time against BSR (10,328,853 dof), and Newton iteration
counts (1.07-1.08M dof).  Every number comes from the logs through results_tables.py's parsers.
"""

import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_tables import (ELEMENTS, operator_name, parse_bench, parse_bench_sweep,  # noqa: E402
                           parse_newton, parse_newton_sweep)

BSR = "#4d4d4d"
BSR32 = "#999999"
P0 = "#d95f02"
QP = ["#1b9e77", "#66a61e", "#7570b3", "#e7298a"]

APPLY = [
    ("BSR f64", "BSR f64", BSR),
    ("BSR f32", "BSR f32", BSR32),
    ("projected P0 f32 packed", "P0 projected (approx.)", P0),
    ("qp f16 blk a32 lanes packed", "qp f16", QP[0]),
    ("qp f16 mean+dev blk a32 lanes packed", "qp f16 mean+dev", QP[1]),
    ("qp i8 mean+dev blk a32 lanes packed", "qp int8 mean+dev", QP[2]),
    ("qp i8/el blk a32 lanes packed", "qp int8 per element", QP[3]),
]


def grouped(ax, values, labels, colors, ylabel, title, fmt="{:.0f}"):
    x = np.arange(len(ELEMENTS))
    w = 0.8 / len(labels)
    for k, (vals, lab, col) in enumerate(zip(values, labels, colors)):
        pos = x - 0.4 + w * (k + 0.5)
        bars = ax.bar(pos, [v if v is not None else 0 for v in vals], w, label=lab, color=col)
        for b, v in zip(bars, vals):
            if v is None:
                ax.text(b.get_x() + b.get_width() / 2, 0, "n/a", ha="center", va="bottom", fontsize=6, rotation=90)
            else:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom",
                        fontsize=6, rotation=90)
    ax.set_xticks(x, ELEMENTS)
    top = max(v for vals in values for v in vals if v is not None)
    ax.set_ylim(0, top * 1.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rows, meta = parse_bench(sorted(glob.glob(os.path.join(src, "bench_*.out"))))
    dof = next(iter(meta.values()))["dof"]

    def series(key, field):
        return [rows.get(el, {}).get(key, {}).get(field) for el in ELEMENTS]

    # 1. application throughput and operator memory
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    grouped(axes[0], [series(k, "mdofs") for k, _, _ in APPLY], [l for _, l, _ in APPLY], [c for _, _, c in APPLY],
            "MDOF/s (higher is better)", f"Application throughput, Grace 72 threads, {dof:,} dof")
    grouped(axes[1], [series(k, "bytes_per_dof") for k, _, _ in APPLY], [l for _, l, _ in APPLY],
            [c for _, _, c in APPLY], "bytes per dof (lower is better)", "Operator memory")
    axes[0].legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "qp_throughput_memory.png"), dpi=160)

    # 2. set-up time per Newton step
    setup = [("BSR f64", "BSR f64 assembly", BSR), ("BSR f32", "BSR f32 assembly", BSR32),
             ("projected P0 f32", "P0 projected", P0), ("qp f16 blk a32 lanes", "qp f16", QP[0]),
             ("qp i8 mean+dev blk a32 lanes", "qp int8 mean+dev", QP[2])]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    grouped(ax, [series(k, "setup_ms") for k, _, _ in setup], [l for _, l, _ in setup], [c for _, _, c in setup],
            "ms per Newton step (lower is better)", f"Set-up of the operator, {dof:,} dof")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "qp_setup.png"), dpi=160)

    # 3. Newton iterations, 1M dof
    newton = {}
    for path in sorted(glob.glob(os.path.join(src, "newton_*_1M_*.out"))):
        runs, m = parse_newton(path)
        newton[m["element"]] = {operator_name(l): r for l, r in runs.items() if r["seconds"] is not None}
    ops = [("BSR f64", "exact (BSR, MF, qp f64)", BSR), ("P0 projected", "P0 projected", P0),
           ("qp f16", "qp f16", QP[0]), ("qp f16md", "qp f16 mean+dev", QP[1]),
           ("qp i8md", "qp int8 mean+dev", QP[2]), ("qp i8", "qp int8 per element", QP[3])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, field, ylabel in ((axes[0], "newton", "Newton iterations"), (axes[1], "linear", "CG iterations")):
        vals = []
        for op, _, _ in ops:
            v = []
            for el in ELEMENTS:
                r = newton.get(el, {}).get(op)
                v.append(r[field] if r else None)
            vals.append(v)
        grouped(ax, vals, [l for _, l, _ in ops], [c for _, _, c in ops], ylabel + " (lower is better)",
                f"{ylabel} to solution, Grace, about 1.08M dof")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "qp_newton_iterations.png"), dpi=160)
    # 4. encoding error, relative energy, warped state, white-noise increment
    err = [("projected P0 f32 packed", "P0 projected", P0), ("qp f16 blk a32 lanes packed", "qp f16", QP[0]),
           ("qp f16 mean+dev blk a32 lanes packed", "qp f16 mean+dev", QP[1]),
           ("qp i8 mean+dev blk a32 lanes packed", "qp int8 mean+dev", QP[2]),
           ("qp i8/el blk a32 lanes packed", "qp int8 per element", QP[3])]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    grouped(ax, [series(k, "energy") for k, _, _ in err], [l for _, l, _ in err], [c for _, _, c in err],
            "relative energy error (lower is better)",
            "Operator error against the exact operator\n(benchmark state: mild warp; white-noise increment)", fmt="{:.0e}")
    ax.set_yscale("log")
    ax.set_ylim(1e-13, 1e-1)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "qp_error.png"), dpi=160)
    sweeps(src, out)
    print("written to", out)


SWEEP_APPLY = [("BSR f64", "BSR f64", BSR, "o-"), ("BSR f32", "BSR f32", BSR32, "o--"),
               ("projected P0 f32 packed", "P0 projected", P0, "s:"),
               ("qp f16 blk a32 lanes packed", "qp f16", QP[0], "^-"),
               ("qp f16 mean+dev blk a32 lanes packed", "qp f16 mean+dev", QP[1], "v-"),
               ("qp i8 mean+dev blk a32 lanes packed", "qp int8 mean+dev", QP[2], "d-")]
SWEEP_NEWTON = [("BSR f64", "BSR f64", BSR, "o-"), ("BSR f32", "BSR f32", BSR32, "o--"),
                ("P0 projected", "P0 projected", P0, "s:"), ("qp f16", "qp f16", QP[0], "^-"),
                ("qp f16md", "qp f16 mean+dev", QP[1], "v-"), ("qp i8md", "qp int8 mean+dev", QP[2], "d-")]


def sweeps(src, out):
    bench = {}
    for path in sorted(glob.glob(os.path.join(src, "bsweep_*.out"))):
        for el, dof, rows in parse_bench_sweep(path):
            bench.setdefault(el, []).append((dof, rows))
    if bench:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=False)
        for ax, el in zip(axes, ELEMENTS):
            pts = sorted(bench.get(el, []))
            for key, lab, col, sty in SWEEP_APPLY:
                xs = [d for d, r in pts if key in r]
                ys = [r[key]["mdofs"] for d, r in pts if key in r]
                if xs:
                    ax.plot(xs, ys, sty, color=col, label=lab, ms=4)
            ax.set_xscale("log")
            ax.set_title(f"{el}: application throughput", fontsize=10)
            ax.set_xlabel("degrees of freedom")
            ax.set_ylabel("MDOF/s (higher is better)")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.suptitle("Grace, 72 threads, packed scatter for the element operators", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "qp_sweep_throughput.png"), dpi=160)
    newton = {}
    for path in sorted(glob.glob(os.path.join(src, "nsweep_*.out"))):
        el = re.search(r"nsweep_(\w+?)_", os.path.basename(path)).group(1).upper()
        newton[el] = parse_newton_sweep(path)
    if newton:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        for ax, el in zip(axes, ELEMENTS):
            pts = sorted(newton.get(el, []), key=lambda t: t[0] or 0)
            for key, lab, col, sty in SWEEP_NEWTON:
                xs, ys = [], []
                for dof, runs in pts:
                    ref = runs.get("BSR f32")
                    r = runs.get(key)
                    if r and ref and r["seconds"] and ref["seconds"]:
                        xs.append(dof)
                        ys.append(r["seconds"] / ref["seconds"])
                if xs:
                    ax.plot(xs, ys, sty, color=col, label=lab, ms=4)
            ax.axhline(1, color=BSR32, lw=0.8)
            ax.set_xscale("log")
            ax.set_title(f"{el}: time to solution / BSR f32", fontsize=10)
            ax.set_xlabel("degrees of freedom")
            ax.set_ylabel("relative time (lower is better)")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.suptitle("Newton, 4 load steps (TET10: 16), Grace, 72 threads", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "qp_sweep_newton.png"), dpi=160)


if __name__ == "__main__":
    main()
