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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "spectral_element_laplacian_bench.csv"
DEFAULT_FIXED_CSV = SCRIPT_DIR / "spectral_element_laplacian_fixed_dofs_bench.csv"

METRICS = {
    "gdof_s_unique": "Full apply unique throughput [GDoF/s]",
    "gdof_s_element": "Full apply element throughput [GDoF/s]",
    "kernel_gdof_s_unique": "Kernel unique throughput [GDoF/s]",
    "kernel_gdof_s_element": "Kernel element throughput [GDoF/s]",
    "cartesian_gdof_s_unique": "Cartesian BXYZE unique throughput [GDoF/s]",
    "cartesian_gdof_s_element": "Cartesian BXYZE element throughput [GDoF/s]",
    "melem_s": "Full apply element rate [MElem/s]",
    "kernel_melem_s": "Kernel element rate [MElem/s]",
    "cartesian_melem_s": "Cartesian BXYZE element rate [MElem/s]",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot grouped bar charts from the SpectralElementLaplacian benchmark CSV."
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            f"CSV path, default: {DEFAULT_CSV}; with --fixed-dofs, "
            f"uses {DEFAULT_FIXED_CSV} if it exists"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR),
        help=f"directory for PNG plots, default: {SCRIPT_DIR}",
    )
    parser.add_argument(
        "--metrics",
        default="gdof_s_unique,kernel_gdof_s_unique",
        help="comma-separated metrics to plot; available: " + ",".join(METRICS),
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="optional cubic mesh size filter, e.g. --size 32",
    )
    parser.add_argument(
        "--total-vs-apply",
        action="store_true",
        help="generate one plot per mesh comparing full total throughput against apply-only throughput",
    )
    parser.add_argument(
        "--global-vs-cellwise",
        action="store_true",
        help="generate plots comparing global gather/scatter apply against cell-wise BXYZE apply+DSS",
    )
    parser.add_argument(
        "--fixed-dofs",
        action="store_true",
        help="generate one plot at approximately fixed global DoFs, selecting rows with level * nx = --fixed-resolution",
    )
    parser.add_argument(
        "--fixed-resolution",
        type=int,
        default=128,
        help="target fine-grid resolution per direction for --fixed-dofs, default: 128",
    )
    return parser.parse_args()


def read_rows(csv_path):
    with open(csv_path, newline="") as stream:
        return list(csv.DictReader(stream))


def default_input_path(fixed_dofs):
    if fixed_dofs and DEFAULT_FIXED_CSV.exists():
        return DEFAULT_FIXED_CSV
    return DEFAULT_CSV


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
        return f"spectral_element_laplacian_{metric}_n{nx}"
    return f"spectral_element_laplacian_{metric}_n{nx}_n{ny}_n{nz}"


def comparison_output_stem(size):
    nx, ny, nz = size
    if nx == ny == nz:
        return f"spectral_element_laplacian_total_vs_apply_n{nx}"
    return f"spectral_element_laplacian_total_vs_apply_n{nx}_n{ny}_n{nz}"


def fixed_dofs_output_stem(resolution):
    return f"spectral_element_laplacian_fixed_dofs_r{resolution}_total_vs_apply"


def global_vs_cellwise_output_stem(size):
    nx, ny, nz = size
    if nx == ny == nz:
        return f"spectral_element_laplacian_global_vs_cellwise_n{nx}"
    return f"spectral_element_laplacian_global_vs_cellwise_n{nx}_n{ny}_n{nz}"


def fixed_dofs_global_vs_cellwise_output_stem(resolution):
    return f"spectral_element_laplacian_fixed_dofs_r{resolution}_global_vs_cellwise"


def plot_total_vs_apply(rows, size, output_dir):
    filtered = [row for row in rows if row_size(row) == size]
    if not filtered:
        return None

    levels = sorted({int(row["level"]) for row in filtered})
    total_values = {int(row["level"]): float(row["gdof_s_unique"]) for row in filtered}
    apply_values = {int(row["level"]): float(row["kernel_gdof_s_unique"]) for row in filtered}

    fig_width = max(7.2, 1.2 * len(levels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)

    centers = list(range(len(levels)))
    width = 0.34

    ax.bar(
        [x - width / 2 for x in centers],
        [total_values[level] for level in levels],
        width=width,
        label="total: zero(y) + apply",
        alpha=0.88,
    )
    ax.bar(
        [x + width / 2 for x in centers],
        [apply_values[level] for level in levels],
        width=width,
        label="apply only",
        alpha=0.88,
    )

    ax.set_title(f"SpectralElementLaplacian total vs apply, mesh {size_label(size)}")
    ax.set_xlabel("Polynomial order p")
    ax.set_ylabel("Unique throughput [GDoF/s]")
    ax.set_xticks(centers)
    ax.set_xticklabels([str(level) for level in levels])
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.legend(loc="upper right", fontsize=9)

    output_path = output_dir / f"{comparison_output_stem(size)}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def plot_fixed_dofs_total_vs_apply(rows, resolution, output_dir):
    filtered = []
    for row in rows:
        nx, ny, nz = row_size(row)
        level = int(row["level"])
        if nx == ny == nz and level * nx == resolution:
            filtered.append(row)

    if not filtered:
        return None

    filtered.sort(key=lambda row: int(row["level"]))
    labels = [f"p={int(row['level'])}\nn={int(row['nx'])}" for row in filtered]
    total_values = [float(row["gdof_s_unique"]) for row in filtered]
    apply_values = [float(row["kernel_gdof_s_unique"]) for row in filtered]
    nunique_values = {int(row["nunique"]) for row in filtered}

    fig_width = max(7.2, 1.35 * len(filtered))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)

    centers = list(range(len(filtered)))
    width = 0.34

    ax.bar(
        [x - width / 2 for x in centers],
        total_values,
        width=width,
        label="total: zero(y) + apply",
        alpha=0.88,
    )
    ax.bar(
        [x + width / 2 for x in centers],
        apply_values,
        width=width,
        label="apply only",
        alpha=0.88,
    )

    if len(nunique_values) == 1:
        dof_text = f"{next(iter(nunique_values)):,} DoFs".replace(",", " ")
    else:
        dof_text = "approximately fixed DoFs"

    ax.set_title(f"SpectralElementLaplacian fixed-DoF comparison, {dof_text}")
    ax.set_xlabel("Polynomial order and macro mesh size")
    ax.set_ylabel("Unique throughput [GDoF/s]")
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.legend(loc="upper right", fontsize=9)

    output_path = output_dir / f"{fixed_dofs_output_stem(resolution)}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def plot_global_vs_cellwise(rows, size, output_dir):
    if not rows or "cellwise_gdof_s_unique" not in rows[0]:
        return None

    filtered = [row for row in rows if row_size(row) == size]
    if not filtered:
        return None

    levels = sorted({int(row["level"]) for row in filtered})
    blocks = sorted({int(row.get("B", 0)) for row in filtered if row.get("B", "") != ""})
    global_values = {}
    cellwise_values = {}
    has_cartesian = "cartesian_gdof_s_unique" in rows[0]
    cellwise_best = {}
    cartesian_best = {}

    for row in filtered:
        level = int(row["level"])
        block = int(row.get("B", 0))
        global_value = float(row["gdof_s_unique"])
        if level not in global_values or global_value > global_values[level]:
            global_values[level] = global_value
        cellwise_values[(level, block)] = float(row["cellwise_gdof_s_unique"])
        cell_value = float(row["cellwise_gdof_s_unique"])
        if level not in cellwise_best or cell_value > cellwise_best[level][0]:
            cellwise_best[level] = (cell_value, block)

        if has_cartesian:
            cart_value = float(row["cartesian_gdof_s_unique"])
            bexec = int(row.get("B_exec", block))
            if level not in cartesian_best or cart_value > cartesian_best[level][0]:
                cartesian_best[level] = (cart_value, block, bexec)

    fig_width = max(7.8, 1.35 * len(levels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)

    centers = list(range(len(levels)))
    n_bars = 3 if has_cartesian else 1 + max(1, len(blocks))
    width = min(0.24, 0.82 / n_bars)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    offset0 = -0.5 * (n_bars - 1) * width
    ax.bar(
        [x + offset0 for x in centers],
        [global_values[level] for level in levels],
        width=width,
        label="global gather/scatter",
        color=color_cycle[0],
        alpha=0.88,
    )

    if has_cartesian:
        ax.bar(
            [x + offset0 + width for x in centers],
            [cellwise_best[level][0] for level in levels],
            width=width,
            label="spectral cell-wise best",
            color=color_cycle[1 % len(color_cycle)],
            alpha=0.88,
        )

        cart_positions = [x + offset0 + 2 * width for x in centers]
        cart_values = [cartesian_best[level][0] for level in levels]
        ax.bar(
            cart_positions,
            cart_values,
            width=width,
            label="cartesian BXYZE best",
            color=color_cycle[2 % len(color_cycle)],
            alpha=0.88,
        )

        for xpos, level, value in zip(cart_positions, levels, cart_values):
            _, bsave, bexec = cartesian_best[level]
            ax.text(xpos, value, f"{bsave}/{bexec}", ha="center", va="bottom", fontsize=8)
    else:
        for ib, block in enumerate(blocks):
            offset = offset0 + (ib + 1) * width
            ax.bar(
                [x + offset for x in centers],
                [cellwise_values.get((level, block), 0.0) for level in levels],
                width=width,
                label=f"cell-wise + DSS, B={block}",
                color=color_cycle[(ib + 1) % len(color_cycle)],
                alpha=0.88,
            )

    ax.set_title(f"SpectralElementLaplacian global vs cell-wise, mesh {size_label(size)}")
    ax.set_xlabel("Polynomial order p")
    ax.set_ylabel("Unique throughput [GDoF/s]")
    ax.set_xticks(centers)
    ax.set_xticklabels([str(level) for level in levels])
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.legend(loc="upper right", fontsize=9)

    output_path = output_dir / f"{global_vs_cellwise_output_stem(size)}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def plot_fixed_dofs_global_vs_cellwise(rows, resolution, output_dir):
    if not rows or "cellwise_gdof_s_unique" not in rows[0]:
        return None

    filtered = []
    for row in rows:
        nx, ny, nz = row_size(row)
        level = int(row["level"])
        if nx == ny == nz and level * nx == resolution:
            filtered.append(row)

    if not filtered:
        return None

    levels = sorted({int(row["level"]) for row in filtered})
    blocks = sorted({int(row.get("B", 0)) for row in filtered if row.get("B", "") != ""})
    global_values = {}
    cellwise_values = {}
    has_cartesian = "cartesian_gdof_s_unique" in rows[0]
    cellwise_best = {}
    cartesian_best = {}
    nunique_values = set()

    for row in filtered:
        level = int(row["level"])
        block = int(row.get("B", 0))
        global_value = float(row["gdof_s_unique"])
        if level not in global_values or global_value > global_values[level]:
            global_values[level] = global_value
        cellwise_values[(level, block)] = float(row["cellwise_gdof_s_unique"])
        cell_value = float(row["cellwise_gdof_s_unique"])
        if level not in cellwise_best or cell_value > cellwise_best[level][0]:
            cellwise_best[level] = (cell_value, block)

        if has_cartesian:
            cart_value = float(row["cartesian_gdof_s_unique"])
            bexec = int(row.get("B_exec", block))
            if level not in cartesian_best or cart_value > cartesian_best[level][0]:
                cartesian_best[level] = (cart_value, block, bexec)

        nunique_values.add(int(row["nunique"]))

    labels = []
    for level in levels:
        matching = [row for row in filtered if int(row["level"]) == level]
        nx = int(matching[0]["nx"])
        labels.append(f"p={level}\nn={nx}")

    fig_width = max(7.8, 1.35 * len(levels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)

    centers = list(range(len(levels)))
    n_bars = 3 if has_cartesian else 1 + max(1, len(blocks))
    width = min(0.24, 0.82 / n_bars)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    offset0 = -0.5 * (n_bars - 1) * width
    ax.bar(
        [x + offset0 for x in centers],
        [global_values[level] for level in levels],
        width=width,
        label="global gather/scatter",
        color=color_cycle[0],
        alpha=0.88,
    )

    if has_cartesian:
        ax.bar(
            [x + offset0 + width for x in centers],
            [cellwise_best[level][0] for level in levels],
            width=width,
            label="spectral cell-wise best",
            color=color_cycle[1 % len(color_cycle)],
            alpha=0.88,
        )

        cart_positions = [x + offset0 + 2 * width for x in centers]
        cart_values = [cartesian_best[level][0] for level in levels]
        ax.bar(
            cart_positions,
            cart_values,
            width=width,
            label="cartesian BXYZE best",
            color=color_cycle[2 % len(color_cycle)],
            alpha=0.88,
        )

        for xpos, level, value in zip(cart_positions, levels, cart_values):
            _, bsave, bexec = cartesian_best[level]
            ax.text(xpos, value, f"{bsave}/{bexec}", ha="center", va="bottom", fontsize=8)
    else:
        for ib, block in enumerate(blocks):
            offset = offset0 + (ib + 1) * width
            ax.bar(
                [x + offset for x in centers],
                [cellwise_values.get((level, block), 0.0) for level in levels],
                width=width,
                label=f"cell-wise + DSS, B={block}",
                color=color_cycle[(ib + 1) % len(color_cycle)],
                alpha=0.88,
            )

    if len(nunique_values) == 1:
        dof_text = f"{next(iter(nunique_values)):,} DoFs".replace(",", " ")
    else:
        dof_text = "approximately fixed DoFs"

    ax.set_title(f"SpectralElementLaplacian global vs cell-wise, {dof_text}")
    ax.set_xlabel("Polynomial order and macro mesh size")
    ax.set_ylabel("Unique throughput [GDoF/s]")
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.legend(loc="upper right", fontsize=9)

    output_path = output_dir / f"{fixed_dofs_global_vs_cellwise_output_stem(resolution)}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def plot_metric(rows, metric, size, output_dir):
    filtered = [row for row in rows if row_size(row) == size]
    if not filtered:
        return None

    levels = sorted({int(row["level"]) for row in filtered})
    values = {int(row["level"]): float(row[metric]) for row in filtered}

    fig_width = max(7.2, 1.2 * len(levels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)

    x = list(range(len(levels)))
    y = [values[level] for level in levels]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax.bar(x, y, width=0.55, color=[colors[i % len(colors)] for i in range(len(levels))], alpha=0.88)
    ax.set_title(f"SpectralElementLaplacian throughput, mesh {size_label(size)}")
    ax.set_xlabel("Polynomial order p")
    ax.set_ylabel(METRICS[metric])
    ax.set_xticks(x)
    ax.set_xticklabels([str(level) for level in levels])
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)

    output_path = output_dir / f"{output_stem(metric, size)}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def main():
    args = parse_args()

    csv_path = Path(args.input).resolve() if args.input else default_input_path(args.fixed_dofs).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    unknown = [metric for metric in metrics if metric not in METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s): {', '.join(unknown)}")

    rows = read_rows(csv_path)

    if args.size is not None:
        sizes = [(args.size, args.size, args.size)]
    else:
        sizes = sorted({row_size(row) for row in rows})

    written = []
    if args.fixed_dofs and args.global_vs_cellwise:
        output_path = plot_fixed_dofs_global_vs_cellwise(rows, args.fixed_resolution, output_dir)
        if output_path:
            written.append(output_path)
    elif args.fixed_dofs:
        output_path = plot_fixed_dofs_total_vs_apply(rows, args.fixed_resolution, output_dir)
        if output_path:
            written.append(output_path)

    if not args.fixed_dofs:
        for size in sizes:
            if args.global_vs_cellwise:
                output_path = plot_global_vs_cellwise(rows, size, output_dir)
                if output_path:
                    written.append(output_path)
            elif args.total_vs_apply:
                output_path = plot_total_vs_apply(rows, size, output_dir)
                if output_path:
                    written.append(output_path)
            else:
                for metric in metrics:
                    output_path = plot_metric(rows, metric, size, output_dir)
                    if output_path:
                        written.append(output_path)
    elif args.total_vs_apply:
        for size in sizes:
            output_path = plot_total_vs_apply(rows, size, output_dir)
            if output_path:
                written.append(output_path)
    elif args.metrics != "gdof_s_unique,kernel_gdof_s_unique" or args.size is not None:
        for size in sizes:
            for metric in metrics:
                output_path = plot_metric(rows, metric, size, output_dir)
                if output_path:
                    written.append(output_path)

    if not written:
        raise RuntimeError("no plots were generated")

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
