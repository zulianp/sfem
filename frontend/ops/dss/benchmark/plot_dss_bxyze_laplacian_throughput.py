#!/usr/bin/env python3

import argparse
import csv
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
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "dss_bxyze_laplacian_bench.csv"
DEFAULT_BASELINE_CSV = SCRIPT_DIR / "dss_bxyze_laplacian_bench_baseline.csv"

METRICS = {
    "gdof_s_unique": "Unique throughput [GDoF/s]",
    "gdof_cell_s": "Cellwise throughput [GDoF/s]",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot grouped bar charts from the BXYZE Cartesian Laplacian benchmark CSV."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_CSV),
        help=f"current CSV path, default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE_CSV),
        help=f"optional baseline CSV path, default: {DEFAULT_BASELINE_CSV}",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="plot only the current CSV, without dashed baseline bars",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR),
        help=f"directory for PNG plots, default: {SCRIPT_DIR}",
    )
    parser.add_argument(
        "--metrics",
        default="gdof_s_unique",
        help="comma-separated metrics to plot; available: " + ",".join(METRICS),
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="optional cubic mesh size filter, e.g. --size 32",
    )
    return parser.parse_args()


def read_rows(csv_path):
    with open(csv_path, newline="") as stream:
        return list(csv.DictReader(stream))


def row_size(row):
    return int(row["nx"]), int(row["ny"]), int(row["nz"])


def size_label(size):
    nx, ny, nz = size
    if nx == ny == nz:
        return f"{nx}^3"
    return f"{nx}x{ny}x{nz}"


def output_stem(metric, size):
    nx, ny, nz = size
    if nx == ny == nz:
        return f"dss_bxyze_laplacian_{metric}_n{nx}"
    return f"dss_bxyze_laplacian_{metric}_n{nx}_n{ny}_n{nz}"


def plot_metric(rows, baseline_rows, metric, size, output_dir):
    filtered = [row for row in rows if row_size(row) == size]
    baseline_filtered = [row for row in baseline_rows if row_size(row) == size]

    levels = sorted({int(row["level"]) for row in filtered} | {int(row["level"]) for row in baseline_filtered})
    blocks = sorted({int(row["B"]) for row in filtered} | {int(row["B"]) for row in baseline_filtered})
    values = {(int(row["level"]), int(row["B"])): float(row[metric]) for row in filtered}
    baseline_values = {(int(row["level"]), int(row["B"])): float(row[metric]) for row in baseline_filtered}

    if not levels or not blocks:
        return None

    fig_width = max(9.0, 1.2 * len(levels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2), constrained_layout=True)

    group_centers = list(range(len(levels)))
    bar_width = min(0.11, 0.82 / max(1, len(blocks)))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ib, block in enumerate(blocks):
        offset = (ib - 0.5 * (len(blocks) - 1)) * bar_width
        x = [center + offset for center in group_centers]
        y = [values.get((level, block), 0.0) for level in levels]
        color = color_cycle[ib % len(color_cycle)]
        ax.bar(x, y, width=bar_width, color=color, alpha=0.86, label=f"B={block}")

        if baseline_values:
            y_base = [baseline_values.get((level, block), 0.0) for level in levels]
            ax.bar(
                x,
                y_base,
                width=bar_width * 0.92,
                facecolor="none",
                edgecolor=color,
                linewidth=1.35,
                linestyle="--",
            )

    ax.set_title(f"BXYZE Cartesian Laplacian throughput, mesh {size_label(size)}")
    ax.set_xlabel("Polynomial order p")
    ax.set_ylabel(METRICS[metric])
    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(level) for level in levels])
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)

    block_legend = ax.legend(title="Block size", ncols=min(len(blocks), 7), fontsize=9, loc="upper left")
    ax.add_artist(block_legend)

    if baseline_values:
        ax.legend(
            handles=[
                Patch(facecolor="0.55", edgecolor="0.55", alpha=0.86, label="current"),
                Patch(facecolor="none", edgecolor="0.25", linestyle="--", linewidth=1.35, label="baseline"),
            ],
            loc="upper right",
            fontsize=9,
        )

    output_path = output_dir / f"{output_stem(metric, size)}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def main():
    args = parse_args()

    csv_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    unknown = [metric for metric in metrics if metric not in METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s): {', '.join(unknown)}")

    rows = read_rows(csv_path)

    baseline_rows = []
    if not args.no_baseline:
        baseline_path = Path(args.baseline).resolve()
        if baseline_path.exists():
            baseline_rows = read_rows(baseline_path)
        else:
            print(f"warning: baseline CSV not found: {baseline_path}")

    if args.size is not None:
        sizes = [(args.size, args.size, args.size)]
    else:
        sizes = sorted({row_size(row) for row in rows} | {row_size(row) for row in baseline_rows})

    written = []
    for size in sizes:
        for metric in metrics:
            output_path = plot_metric(rows, baseline_rows, metric, size, output_dir)
            if output_path:
                written.append(output_path)

    if not written:
        raise RuntimeError("no plots were generated")

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
