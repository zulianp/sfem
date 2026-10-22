#!/usr/bin/env python3

"""Compare throughput of independently optimized cellwise apply/apply+DSS with global8."""

import argparse
import csv
import math
import os
import statistics
import tempfile
from pathlib import Path

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "sfem_dss_matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "dss_bxyze_affine_laplacian_global_bench.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_CSV), help="full benchmark CSV, not the _best summary")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR), help="directory for PNG plots")
    parser.add_argument("--size", type=int, help="optional cubic macro mesh size filter, e.g. 16")
    parser.add_argument("--threads", type=int, help="optional thread count filter, e.g. 8")
    parser.add_argument("--statistic", choices=("median", "best"), default="median",
                        help="median or minimum timings; default: median")
    return parser.parse_args()


def row_size(row):
    return int(row["nx"]), int(row["ny"]), int(row["nz"])


def select_comparisons(rows, statistic):
    suffix = "median_s" if statistic == "median" else "min_s"
    keys = [f"{method}_tts_{suffix}" for method in ("global8", "cellwise_apply", "cellwise_apply_dss")]
    by_degree = {}
    seen = set()
    for row in rows:
        degree = int(row["degree"])
        configuration = degree, int(row["B_save"]), int(row["B_exec"])
        if configuration in seen:
            raise ValueError(f"duplicate configuration {configuration}; use the full benchmark CSV")
        seen.add(configuration)
        if any(not math.isfinite(float(row[key])) or float(row[key]) <= 0 for key in keys):
            raise ValueError(f"timings must be finite and positive for {configuration}")
        by_degree.setdefault(degree, []).append(row)

    selected = []
    for degree, group in sorted(by_degree.items()):
        dofs = {int(row["n_global_dofs"]) for row in group}
        if len(dofs) != 1 or min(dofs) <= 0:
            raise ValueError(f"inconsistent global DoF count for p={degree}")
        fixed = any(row.get("chosen_for") for row in group)
        apply_group = [row for row in group if row.get("chosen_for") in ("apply", "both")] if fixed else group
        dss_group = [row for row in group if row.get("chosen_for") in ("apply_dss", "both")] if fixed else group
        if not apply_group or not dss_group:
            raise ValueError(f"missing preselected B for p={degree}")
        apply_row = min(apply_group, key=lambda row: float(row[keys[1]]))
        dss_row = min(dss_group, key=lambda row: float(row[keys[2]]))
        # The global implementation does not depend on B, but is measured once
        # per configuration. Use every recorded baseline, independently of the
        # cellwise winners: median of medians, or minimum of minima. Ratios are
        # recomputed against this single global bar, not copied from paired CSV
        # speedups, which can use different global timings for the two winners.
        global_times = [float(row[keys[0]]) for row in group]
        selected.append({
            "degree": degree,
            "dofs": dofs.pop(),
            "global_time": statistics.median(global_times) if statistic == "median" else min(global_times),
            "apply_time": float(apply_row[keys[1]]),
            "dss_time": float(dss_row[keys[2]]),
            "apply_block": (int(apply_row["B_save"]), int(apply_row["B_exec"])),
            "dss_block": (int(dss_row["B_save"]), int(dss_row["B_exec"])),
        })
    return selected


def output_stem(size, threads, statistic, indexing, scatter_policy):
    nx, ny, nz = size
    mesh = f"n{nx}" if nx == ny == nz else f"n{nx}_n{ny}_n{nz}"
    variant = "" if indexing == "elements" else f"_{indexing}"
    if scatter_policy == "boundary":
        variant += "_boundary_atomic"
    return f"dss_bxyze_affine_laplacian_global_gdof_s_unique_{mesh}_t{threads}{variant}_{statistic}"


def plot_comparison(selected, size, threads, statistic, output_dir, indexing, scatter_policy, fixed_b):
    def values(time_key):
        return [1e-9 * row["dofs"] / row[time_key] for row in selected]

    global_values, apply_values, dss_values = [values(key) for key in ("global_time", "apply_time", "dss_time")]
    fig, ax = plt.subplots(figsize=(9.0, 5.3), constrained_layout=True)
    centers = list(range(len(selected)))
    width = 0.24
    scatter_label = ("boundary atomics" if scatter_policy == "boundary" else "atomic scatter") if threads > 1 else "scatter"
    global_label = (f"global: structured IDs + {scatter_label}"
                    if indexing == "structured" else f"global: gather + apply + {scatter_label}")
    ax.bar([x - width for x in centers], global_values, width=width,
           label=global_label, color="tab:blue", alpha=0.88)
    apply_bars = ax.bar(centers, apply_values, width=width,
                        label="cellwise: apply (best B)", color="tab:orange", alpha=0.88)
    dss_bars = ax.bar([x + width for x in centers], dss_values, width=width,
                      label="cellwise: apply + DSS (best B)", color="tab:green", alpha=0.88)
    for bars, time_key in ((apply_bars, "apply_time"), (dss_bars, "dss_time")):
        for bar, row in zip(bars, selected):
            speedup = row["global_time"] / row[time_key]
            ax.annotate(f"{speedup:.2f} X",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    nx, ny, nz = size
    mesh_label = rf"${nx}^3$" if nx == ny == nz else f"{nx}x{ny}x{nz}"
    thread_label = "thread" if threads == 1 else "threads"
    timing_label = "Median" if statistic == "median" else "Minimum"
    selection_note = ("B preselected from prior sweep" if fixed_b
                      else "best B selected separately for apply and apply+DSS")
    indexing_note = "; structured global IDs" if indexing == "structured" else ""
    ax.set_title(f"Affine Laplacian throughput, mesh {mesh_label}, {threads} {thread_label}\n"
                 f"{timing_label} timings; {selection_note}{indexing_note}")
    ax.set_xlabel("Polynomial order p\nB_save/B_exec: apply | apply+DSS")
    ax.set_ylabel("Unique throughput [GDoF/s]")
    ax.set_xticks(centers)
    ax.set_xticklabels([
        f"{row['degree']}\n{row['apply_block'][0]}/{row['apply_block'][1]} | "
        f"{row['dss_block'][0]}/{row['dss_block'][1]}"
        for row in selected
    ])
    ax.set_ylim(0, 1.42 * max(global_values + apply_values + dss_values))
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.legend(loc="upper left", fontsize=9)
    baseline_note = "median of recorded medians" if statistic == "median" else "minimum of recorded minima"
    fig.supxlabel(f"Global: {baseline_note} across B configurations; labels = speedup vs global", fontsize=8)
    output_path = output_dir / f"{output_stem(size, threads, statistic, indexing, scatter_policy)}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    with open(Path(args.input).resolve(), newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.size is not None:
        rows = [row for row in rows if row_size(row) == (args.size,) * 3]
    if args.threads is not None:
        rows = [row for row in rows if int(row["threads"]) == args.threads]
    if not rows:
        raise RuntimeError("no benchmark rows match the requested filters")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted({(row_size(row), int(row["threads"])) for row in rows})
    for size, threads in groups:
        filtered = [row for row in rows if row_size(row) == size and int(row["threads"]) == threads]
        indexing_modes = {row.get("global_indexing") or "elements" for row in filtered}
        if len(indexing_modes) != 1:
            raise ValueError("mixed global indexing modes in one plot")
        indexing = indexing_modes.pop()
        if indexing not in ("elements", "structured"):
            raise ValueError(f"unsupported global indexing mode: {indexing}")
        scatter_policies = {row.get("global_scatter_policy") or "all" for row in filtered}
        if len(scatter_policies) != 1:
            raise ValueError("mixed global scatter policies in one plot")
        scatter_policy = scatter_policies.pop()
        if scatter_policy not in ("all", "boundary"):
            raise ValueError(f"unsupported global scatter policy: {scatter_policy}")
        fixed_b = any(row.get("chosen_for") for row in filtered)
        selected = select_comparisons(filtered, args.statistic)
        print(plot_comparison(selected, size, threads, args.statistic, output_dir, indexing, scatter_policy, fixed_b))


if __name__ == "__main__":
    main()
