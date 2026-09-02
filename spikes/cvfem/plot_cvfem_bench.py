#!/usr/bin/env python3
"""Publication-quality figures from a CVFEM benchmark CSV.

    python3 plot_cvfem_bench.py bench.csv -o plots/
    python3 plot_cvfem_bench.py bench.csv -o plots/ --format pdf --dark

Needs matplotlib. If you only want a summary to read or share, use
report_cvfem_bench.py instead -- that one is standard library only and draws its
own SVG, so it works inside a bare uenv.

Figures written:
    layouts.<ext>     throughput by layout, one panel per operation
    packsize.<ext>    assembly and residual vs pack size
    threads.<ext>     thread scaling, with an ideal-scaling reference
    size.<ext>        throughput vs problem size, with the cache-resident region marked
    kernels.<ext>     element kernel variants
    breakdown.<ext>   assembly phase budget, stacked
"""

import argparse
import os
import sys
from collections import OrderedDict, defaultdict

try:
    import matplotlib
except ImportError:
    sys.exit("matplotlib is not available.\n"
             "Either `pip install matplotlib`, or use report_cvfem_bench.py, which needs nothing.")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.lines  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_cvfem_bench import Bench, LAYOUT_ORDER, OP_ORDER, OP_LABEL, PHASE_LABEL  # noqa: E402

# The same validated categorical palette the HTML report uses, so figures dropped
# into a paper and the shared page read as one set. Light and dark are separate
# steps against their own surface, not an inversion.
PALETTE = {
    "light": {
        # four categorical hues for the layouts, Okabe-Ito derived, all-pairs safe
        "series": ["#0072B2", "#D55E00", "#009E73", "#CC79A7"],
        # six ordinal steps for the assembly phases: they are a pipeline, not
        # categories, so a single-hue ramp is both correct and CVD-safe
        "phase": ["#83B6D6", "#63A0C9", "#4283B2", "#276492", "#144870", "#092E4E"],
        "ink": "#191B1A", "ink2": "#525854", "muted": "#858B86",
        "rule": "#DDDFD8", "surface": "#FCFCFB",
    },
    "dark": {
        "series": ["#2A8CC8", "#D46A1C", "#08A176", "#C874A4"],
        "phase": ["#1E5580", "#2E6E9C", "#4189B8", "#5EA4CE", "#87BFDF", "#B4D8EE"],
        "ink": "#E7E9E4", "ink2": "#AEB4AE", "muted": "#838A84",
        "rule": "#2B302D", "surface": "#1A1D1B",
    },
}

# Secondary encoding, so a layout stays identifiable without colour. Required:
# inside the dark lightness band the blue/pink pair sits just under the
# colour-vision separation floor.
DASHES = ["-", (0, (6, 3)), (0, (2, 3)), (0, (8, 3, 2, 3))]
MARKERS = ["o", "s", "^", "D"]


def dash_for(name, order):
    try:
        i = order.index(name)
    except ValueError:
        i = len(order)
    return DASHES[i % len(DASHES)], MARKERS[i % len(MARKERS)]


def style(theme):
    p = PALETTE[theme]
    plt.rcParams.update({
        "figure.facecolor": p["surface"],
        "axes.facecolor": p["surface"],
        "savefig.facecolor": p["surface"],
        "text.color": p["ink"],
        "axes.labelcolor": p["ink2"],
        "axes.edgecolor": p["rule"],
        "xtick.color": p["muted"],
        "ytick.color": p["muted"],
        "grid.color": p["rule"],
        "axes.grid": True,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "font.size": 10,
        "font.family": ["DejaVu Sans"],
        "figure.dpi": 140,
    })
    return p


def color_for(name, order, pal):
    try:
        i = order.index(name)
    except ValueError:
        i = len(order)
    return pal["series"][i % len(pal["series"])]


def save(fig, out_dir, name, ext):
    path = os.path.join(out_dir, "%s.%s" % (name, ext))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  %s" % path)
    return path


# ------------------------------------------------------------------ the figures

def fig_layouts(bench, pal, threads, out_dir, ext):
    ops = [o for o in OP_ORDER if bench.select(sweep="layout", operation=o, threads=threads)]
    if not ops:
        return
    fig, axes = plt.subplots(1, len(ops), figsize=(3.2 * len(ops), 3.1), squeeze=False)
    for ax, op in zip(axes[0], ops):
        rows = sorted(
            bench.select(sweep="layout", operation=op, threads=threads),
            key=lambda r: LAYOUT_ORDER.index(r["layout"]) if r["layout"] in LAYOUT_ORDER else 9,
        )
        names = [r["layout"] for r in rows]
        vals = [r["MDOF_s"] for r in rows]
        cols = [color_for(n, LAYOUT_ORDER, pal) for n in names]
        bars = ax.barh(range(len(rows)), vals, color=cols, height=0.62)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_title(OP_LABEL.get(op, op))
        ax.set_xlabel("MDOF/s")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.28)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() * 1.03, b.get_y() + b.get_height() / 2, "%.1f" % v,
                    va="center", fontsize=8.5, color=pal["ink2"])
    fig.suptitle("Throughput by assembly layout (%d threads)" % threads, y=1.04, fontsize=12)
    save(fig, out_dir, "layouts", ext)


def fig_packsize(bench, pal, threads, out_dir, ext):
    ops = [o for o in ("assemble", "residual") if bench.select(sweep="packsize", operation=o, threads=threads)]
    if not ops:
        return
    fig, axes = plt.subplots(1, len(ops), figsize=(4.4 * len(ops), 3.3), squeeze=False)
    for ax, op in zip(axes[0], ops):
        for lay in LAYOUT_ORDER:
            pts = sorted((r["pack_size"], r["MDOF_s"])
                         for r in bench.select(sweep="packsize", operation=op, layout=lay, threads=threads))
            if not pts:
                continue
            ls, mk = dash_for(lay, LAYOUT_ORDER)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle=ls, marker=mk,
                    color=color_for(lay, LAYOUT_ORDER, pal), label=lay)
            best = max(pts, key=lambda p: p[1])
            ax.annotate("%d" % best[0], best, textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=pal["muted"])
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("pack size (elements)")
        ax.set_ylabel("MDOF/s")
        ax.set_title(OP_LABEL.get(op, op))
        ax.legend()
    fig.suptitle("Pack size sweep (%d threads); labels mark each layout's optimum" % threads,
                 y=1.03, fontsize=12)
    save(fig, out_dir, "packsize", ext)


def fig_threads(bench, pal, out_dir, ext):
    ops = [o for o in ("assemble", "residual") if bench.select(sweep="threads", operation=o)]
    if not ops:
        return
    fig, axes = plt.subplots(1, len(ops), figsize=(4.4 * len(ops), 3.3), squeeze=False)
    for ax, op in zip(axes[0], ops):
        any_ideal = False
        for lay in LAYOUT_ORDER:
            pts = sorted((r["threads"], r["MDOF_s"])
                         for r in bench.select(sweep="threads", operation=op, layout=lay))
            if not pts:
                continue
            col = color_for(lay, LAYOUT_ORDER, pal)
            ls, mk = dash_for(lay, LAYOUT_ORDER)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle=ls, marker=mk,
                    color=col, label=lay)
            # Each layout gets its own ideal reference, anchored to its own smallest
            # thread count. A single shared reference would let a faster layout appear
            # to scale better than perfectly, which is just the anchor being wrong.
            t0, v0 = pts[0]
            ts = [p[0] for p in pts]
            ax.plot(ts, [v0 * t / t0 for t in ts], "--", color=col, linewidth=1.0,
                    alpha=0.35, zorder=0)
            any_ideal = True
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("threads")
        ax.set_ylabel("MDOF/s")
        ax.set_title(OP_LABEL.get(op, op))
        handles, labels = ax.get_legend_handles_labels()
        if any_ideal:
            handles.append(matplotlib.lines.Line2D([], [], color=pal["muted"], linestyle="--",
                                                   alpha=0.6, linewidth=1.0))
            labels.append("ideal (per layout)")
        ax.legend(handles, labels)
    fig.suptitle("Thread scaling; the gap to each layout's own dashed ideal is bandwidth contention",
                 y=1.03, fontsize=12)
    save(fig, out_dir, "threads", ext)


def fig_size(bench, pal, threads, out_dir, ext, cache_mib=None):
    ops = [o for o in ("assemble", "residual") if bench.select(sweep="size", operation=o, threads=threads)]
    if not ops:
        return
    fig, axes = plt.subplots(1, len(ops), figsize=(4.4 * len(ops), 3.3), squeeze=False)
    for ax, op in zip(axes[0], ops):
        for lay in LAYOUT_ORDER:
            pts = sorted((r["dofs"], r["MDOF_s"], r["bsr_values_MiB"])
                         for r in bench.select(sweep="size", operation=op, layout=lay, threads=threads))
            if not pts:
                continue
            ls, mk = dash_for(lay, LAYOUT_ORDER)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle=ls, marker=mk,
                    color=color_for(lay, LAYOUT_ORDER, pal), label=lay)
        if cache_mib and op == "assemble":
            # first size whose matrix no longer fits the last-level cache
            edge = None
            for r in sorted(bench.select(sweep="size", operation=op, threads=threads),
                            key=lambda r: r["dofs"]):
                if r["bsr_values_MiB"] > cache_mib:
                    edge = r["dofs"]
                    break
            if edge:
                ax.axvline(edge, color=pal["muted"], linestyle=":", linewidth=1.2)
                ax.annotate("matrix exceeds\n%d MiB cache" % cache_mib, (edge, ax.get_ylim()[1]),
                            textcoords="offset points", xytext=(6, -22), fontsize=8,
                            color=pal["muted"])
        ax.set_xscale("log")
        ax.set_xlabel("dofs")
        ax.set_ylabel("MDOF/s")
        ax.set_title(OP_LABEL.get(op, op))
        ax.legend()
    fig.suptitle("Problem-size scaling (%d threads)" % threads, y=1.03, fontsize=12)
    save(fig, out_dir, "size", ext)


def fig_kernels(bench, pal, threads, out_dir, ext):
    rows = bench.select(sweep="kernel", threads=threads)
    if not rows:
        return
    ops = [o for o in OP_ORDER if any(r["operation"] == o for r in rows)]
    fig, axes = plt.subplots(1, len(ops), figsize=(3.6 * len(ops), 3.2), squeeze=False)
    for ax, op in zip(axes[0], ops):
        sub = [r for r in rows if r["operation"] == op]
        kernels = sorted({r["kernel"] for r in sub})
        layouts = sorted({r["layout"] for r in sub},
                         key=lambda l: LAYOUT_ORDER.index(l) if l in LAYOUT_ORDER else 9)
        w = 0.8 / max(len(layouts), 1)
        for j, lay in enumerate(layouts):
            vals = []
            for k in kernels:
                m = [r["MDOF_s"] for r in sub if r["kernel"] == k and r["layout"] == lay]
                vals.append(max(m) if m else 0.0)
            ax.bar([i + j * w for i in range(len(kernels))], vals, width=w * 0.9,
                   color=color_for(lay, LAYOUT_ORDER, pal), label=lay)
        ax.set_xticks([i + w * (len(layouts) - 1) / 2 for i in range(len(kernels))])
        ax.set_xticklabels(kernels, rotation=20, ha="right")
        ax.set_ylabel("MDOF/s")
        ax.set_title(OP_LABEL.get(op, op))
        ax.grid(axis="x", visible=False)
        ax.legend()
    fig.suptitle("Element kernel variants (%d threads)" % threads, y=1.04, fontsize=12)
    save(fig, out_dir, "kernels", ext)


def fig_breakdown(bench, pal, threads, out_dir, ext):
    rows = sorted(bench.select(sweep="breakdown", operation="assemble", threads=threads),
                  key=lambda r: LAYOUT_ORDER.index(r["layout"]) if r["layout"] in LAYOUT_ORDER else 9)
    if not rows:
        return
    keys = [k for k in PHASE_LABEL if any(r[k] == r[k] and r[k] > 0 for r in rows)]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(7.6, 0.85 * len(rows) + 2.4))
    left = [0.0] * len(rows)
    for j, k in enumerate(keys):
        vals = [(r[k] if r[k] == r[k] else 0.0) for r in rows]
        ax.barh(range(len(rows)), vals, left=left, height=0.55,
                color=pal["phase"][j % len(pal["phase"])], label=PHASE_LABEL[k],
                edgecolor=pal["surface"], linewidth=1.2)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["%s\npack %d" % (r["layout"], r["pack_size"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("milliseconds per assemble, summed over %d threads" % threads)
    ax.grid(axis="y", visible=False)
    for i, tot in enumerate(left):
        ax.text(tot * 1.01, i, "%.0f" % tot, va="center", fontsize=8.5, color=pal["ink2"])
    ax.set_xlim(0, max(left) * 1.12)
    ax.legend(ncol=min(len(keys), 3), loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Where an assemble goes")
    save(fig, out_dir, "breakdown", ext)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="plots")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dark", action="store_true", help="render against the dark surface")
    ap.add_argument("--threads", type=int, default=None,
                    help="thread count for the fixed-width figures (default: the maximum measured)")
    ap.add_argument("--cache-mib", type=float, default=None,
                    help="last-level cache size, to mark the cache-resident region (Grace: 117)")
    args = ap.parse_args()

    bench = Bench.load(args.csv)
    pal = style("dark" if args.dark else "light")
    threads = args.threads or max(r["threads"] for r in bench.rows)
    os.makedirs(args.out, exist_ok=True)

    print("plots from %s (%d configurations, %d threads):" % (args.csv, len(bench.best), threads))
    fig_layouts(bench, pal, threads, args.out, args.format)
    fig_packsize(bench, pal, threads, args.out, args.format)
    fig_threads(bench, pal, args.out, args.format)
    fig_size(bench, pal, threads, args.out, args.format, args.cache_mib)
    fig_kernels(bench, pal, threads, args.out, args.format)
    fig_breakdown(bench, pal, threads, args.out, args.format)


if __name__ == "__main__":
    main()
