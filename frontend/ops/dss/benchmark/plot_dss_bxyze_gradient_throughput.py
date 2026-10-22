#!/usr/bin/env python3

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "sfem_dss_matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "dss_bxyze_gradient_bench.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot global9, global8 and cellwise gradient throughput, with speedup labels. "
            "Select the fastest cellwise configuration per order, mesh and thread count; "
            "use its paired global measurements for comparison."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_CSV), help=f"CSV path, default: {DEFAULT_CSV}")
    parser.add_argument(
        "--output-dir", default=str(SCRIPT_DIR), help=f"directory for PNG plots, default: {SCRIPT_DIR}"
    )
    parser.add_argument("--size", type=int, help="optional cubic macro mesh size filter, e.g. 16")
    parser.add_argument("--threads", type=int, help="optional thread count filter, e.g. 8")
    parser.add_argument("--b-save", type=int, help="optional storage block size filter")
    parser.add_argument("--b-exec", type=int, help="optional execution batch size filter")
    parser.add_argument(
        "--statistic", choices=("median", "best"), default="median",
        help="use median timings or minimum timings (best), default: median",
    )
    parser.add_argument(
        "--block-analysis", action="store_true",
        help="also plot B_save/B_exec throughput maps for every order, mesh and thread count",
    )
    return parser.parse_args()


def read_rows(csv_path):
    with open(csv_path, newline="") as stream:
        return list(csv.DictReader(stream))


def row_size(row):
    return int(row["nx"]), int(row["ny"]), int(row["nz"])


def size_label(size):
    nx, ny, nz = size
    return f"{nx}^3" if nx == ny == nz else f"{nx}x{ny}x{nz}"


def select_best_rows(rows, statistic):
    time_key = "cellwise_tts_median_s" if statistic == "median" else "cellwise_tts_min_s"
    selected = {}
    for row in rows:
        degree = int(row["degree"])
        elapsed = float(row[time_key])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"invalid cellwise timing for p={degree}")
        if degree not in selected or elapsed < float(selected[degree][time_key]):
            selected[degree] = row
    return [selected[degree] for degree in sorted(selected)]


def output_stem(size, threads, args):
    nx, ny, nz = size
    mesh = f"n{nx}" if nx == ny == nz else f"n{nx}_n{ny}_n{nz}"
    stem = f"dss_bxyze_gradient_gdof_s_unique_{mesh}_t{threads}_{args.statistic}"
    if args.b_save is not None:
        stem += f"_bs{args.b_save}"
    if args.b_exec is not None:
        stem += f"_be{args.b_exec}"
    return stem


def plot_throughput(rows, size, threads, output_dir, args):
    selected = select_best_rows(rows, args.statistic)
    degrees = [int(row["degree"]) for row in selected]
    suffix = "median_s" if args.statistic == "median" else "min_s"
    global_times = [float(row[f"global_tts_{suffix}"]) for row in selected]
    global8_times = [float(row[f"global8_tts_{suffix}"]) for row in selected]
    cellwise_times = [float(row[f"cellwise_tts_{suffix}"]) for row in selected]
    if any(not math.isfinite(value) or value <= 0 for value in global_times + global8_times):
        raise ValueError("global timings must be finite and positive")

    # All bars and both ratios come from the same three-method measurement.
    dofs = [1e-9 * int(row["n_global_dofs"]) for row in selected]
    global_values = [n / t for n, t in zip(dofs, global_times)]
    global8_values = [n / t for n, t in zip(dofs, global8_times)]
    cellwise_values = [n / t for n, t in zip(dofs, cellwise_times)]
    speedups = [g / c for g, c in zip(global_times, cellwise_times)]
    speedups_vs_global8 = [g / c for g, c in zip(global8_times, cellwise_times)]

    fig_width = max(7.8, 1.35 * len(degrees))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)
    centers = list(range(len(degrees)))
    width = 0.24
    ax.bar(
        [x - width for x in centers], global_values,
        width=width, label="global: 9 contractions", alpha=0.88,
    )
    ax.bar(
        centers, global8_values,
        width=width, label="global: 8 contractions", alpha=0.88,
    )
    cellwise_bars = ax.bar(
        [x + width for x in centers], cellwise_values,
        width=width, label="cellwise BXYZE: 8 contractions", alpha=0.88,
    )
    for bar, speedup, speedup8 in zip(cellwise_bars, speedups, speedups_vs_global8):
        ax.annotate(
            f"{speedup:.2f} X vs 9\n{speedup8:.2f} X vs 8",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    timing_label = "Median" if args.statistic == "median" else "Minimum"
    thread_label = "thread" if threads == 1 else "threads"
    ax.set_title(
        f"BXYZE gradient throughput, mesh {size_label(size)}, {threads} {thread_label}\n"
        f"{timing_label} timings; best cellwise configuration per order"
    )
    ax.set_xlabel("Polynomial order p\nB_save / B_exec (cellwise)")
    ax.set_ylabel("Unique throughput [GDoF/s]")
    ax.set_xticks(centers)
    ax.set_xticklabels([
        f"{degree}\n{int(row['B_save'])}/{int(row['B_exec'])}"
        for degree, row in zip(degrees, selected)
    ])
    ax.set_ylim(0, 1.42 * max(global_values + global8_values + cellwise_values))
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.legend(loc="upper left", fontsize=9)

    output_path = output_dir / f"{output_stem(size, threads, args)}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_block_analysis(rows, size, threads, output_dir, args):
    degrees = sorted({int(row["degree"]) for row in rows})
    blocks = sorted({int(row["B_save"]) for row in rows})
    batches = sorted({int(row["B_exec"]) for row in rows})
    time_suffix = "median_s" if args.statistic == "median" else "min_s"
    time_key = f"cellwise_tts_{time_suffix}"
    values = {}
    for row in rows:
        elapsed = float(row[time_key])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("cellwise timings must be finite and positive")
        key = int(row["degree"]), int(row["B_save"]), int(row["B_exec"])
        if key in values:
            raise ValueError(f"duplicate block configuration: {key}")
        values[key] = 1e-9 * int(row["n_global_dofs"]) / elapsed

    maximum = max(values.values())
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.93")
    fig, axes = plt.subplots(
        1, len(degrees), figsize=(max(4.0, 3.3 * len(degrees)), 3.8),
        squeeze=False, sharey=True, constrained_layout=True,
    )
    for ax, degree in zip(axes[0], degrees):
        matrix = [[values.get((degree, bs, be), math.nan) for bs in blocks] for be in batches]
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=maximum)
        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels(blocks, fontsize=12)
        ax.set_yticks(range(len(batches)))
        ax.set_yticklabels(batches, fontsize=12)
        ax.set_xlabel(r"$B_{\mathrm{save}}$", fontsize=13)
        ax.set_title(f"p={degree}", fontsize=13)
        best = max((value, bs, be) for (p, bs, be), value in values.items() if p == degree)
        for iy, be in enumerate(batches):
            for ix, bs in enumerate(blocks):
                value = matrix[iy][ix]
                label = f"{value:.3f}" if math.isfinite(value) else "--"
                color = "white" if math.isfinite(value) and value / maximum < 0.55 else "black"
                ax.text(ix, iy, label, ha="center", va="center", color=color, fontsize=11.5)
                if (bs, be) == best[1:]:
                    ax.add_patch(Rectangle(
                        (ix - 0.48, iy - 0.48), 0.96, 0.96,
                        fill=False, edgecolor="tab:red", linewidth=2.0,
                    ))
    axes[0][0].set_ylabel(r"$B_{\mathrm{exec}}$", fontsize=13)
    colorbar = fig.colorbar(image, ax=list(axes[0]), fraction=0.035, pad=0.025)
    colorbar.set_label("Unique throughput [GDoF/s]", fontsize=12)
    colorbar.ax.tick_params(labelsize=11)
    timing_label = "Median" if args.statistic == "median" else "Minimum"
    thread_label = "thread" if threads == 1 else "threads"
    fig.suptitle(
        f"BXYZE gradient block sizes, mesh {size_label(size)}, {threads} {thread_label}\n"
        f"{timing_label} timings; red outline = best per order; gray = unmeasured",
        fontsize=13,
    )
    output_path = output_dir / f"{output_stem(size, threads, args)}_blocks.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    rows = read_rows(Path(args.input).resolve())
    if args.size is not None:
        rows = [row for row in rows if row_size(row) == (args.size,) * 3]
    for argument, column in ((args.threads, "threads"), (args.b_save, "B_save"), (args.b_exec, "B_exec")):
        if argument is not None:
            rows = [row for row in rows if int(row[column]) == argument]
    if not rows:
        raise RuntimeError("no benchmark rows match the requested filters")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted({(row_size(row), int(row["threads"])) for row in rows})
    for size, threads in groups:
        filtered = [row for row in rows if row_size(row) == size and int(row["threads"]) == threads]
        print(plot_throughput(filtered, size, threads, output_dir, args))
        if args.block_analysis:
            print(plot_block_analysis(filtered, size, threads, output_dir, args))


if __name__ == "__main__":
    main()
