#!/usr/bin/env python3
"""Plot fixed-iteration CG throughput for four meshes in one figure."""

import argparse
import csv
import math
import os
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "dss_bxyze_cg_fixed_budget_n8_16_32_64_t1_summary.csv"
DEFAULT_OUTPUT = HERE / "dss_bxyze_cg_fixed_budget_throughput_t1.png"
SIZES = (8, 16, 32, 64)
DEGREES = (2, 4, 8)


def read_cases(path, threads):
    cases = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            size, degree = int(row["size"]), int(row["degree"])
            key = size, degree
            if key in cases or int(row["threads"]) != threads or row["status"] != "complete":
                raise ValueError(f"duplicate, wrong-thread or incomplete case {key}")
            count = int(row["fixed_iterations"])
            if count < 1 or any(int(row[f"{m}_completed"]) != 2 for m in ("global", "cellwise")):
                raise ValueError(f"invalid iteration count or repetition set for {key}")
            dofs = (size * degree + 1) ** 3
            times = [float(row[f"{m}_attempt_seconds_median"]) for m in ("global", "cellwise")]
            if any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError(f"invalid solve time for {key}")
            cases[key] = count, [1e-9 * count * dofs / t for t in times], times[0] / times[1]
    expected = {(size, degree) for size in SIZES for degree in DEGREES}
    if set(cases) != expected:
        raise ValueError(f"expected 12 cases; missing={sorted(expected - set(cases))}, extra={sorted(set(cases) - expected)}")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--global-label", default="Global")
    parser.add_argument("--reused-cellwise", action="store_true")
    args = parser.parse_args()
    cases = read_cases(args.input, args.threads)

    cache = Path(tempfile.gettempdir()) / "sfem_dss_matplotlib_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(9.2, 12.0), sharey=True, constrained_layout=True)
    width = 0.32
    maximum = max(value for count, pair, speedup in cases.values() for value in pair)
    for ax, size in zip(axes, SIZES):
        global_values = [cases[size, p][1][0] for p in DEGREES]
        cellwise_values = [cases[size, p][1][1] for p in DEGREES]
        ax.bar([i - width / 2 for i in range(3)], global_values, width, color="tab:blue", alpha=.88,
               label=args.global_label if size == SIZES[0] else None)
        bars = ax.bar([i + width / 2 for i in range(3)], cellwise_values, width, color="tab:green", alpha=.88,
                      label="Cellwise + DSS" if size == SIZES[0] else None)
        for p, bar in zip(DEGREES, bars):
            ax.annotate(f"{cases[size, p][2]:.2f}×",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=10)
        ax.set_title(rf"Mesh ${size}^3$", loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks(range(3), [f"p={p}\nK={cases[size, p][0]}" for p in DEGREES])
        ax.set_xlim(-.55, 2.55)
        ax.set_ylim(0, maximum * 1.28)
        ax.grid(axis="y", linestyle=":", linewidth=.8, alpha=.55)
        ax.set_axisbelow(True)
    fig.suptitle(f"Matrix-free CG throughput · {args.threads} thread{'s' if args.threads != 1 else ''}",
                 fontsize=15, fontweight="bold")
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.03), ncol=2, frameon=False)
    fig.supylabel("Unique global DoF × CG iterations / solve time [GDoF/s]")
    provenance = " · cellwise samples reused" if args.reused_cellwise else ""
    fig.supxlabel("Median of 2 runs per method · fixed K for each mesh and order · labels show cellwise speedup" + provenance,
                 fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
