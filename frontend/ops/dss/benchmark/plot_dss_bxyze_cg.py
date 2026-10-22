#!/usr/bin/env python3
"""Plot complete CG campaign comparisons, retaining placeholders for pending cases."""

import argparse
from contextlib import contextmanager
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "dss_bxyze_cg_16_64_bench_summary.csv"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV, help="campaign _summary.csv")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--size", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--completed-only", action="store_true",
                        help="show only complete validated configurations, without pending placeholders")
    parser.add_argument("--metrics", nargs="+", choices=("tts", "throughput", "iterations"),
                        default=["throughput"])
    parser.add_argument("--time-scale", choices=("linear", "log"), default="log")
    parser.add_argument("--between-solves", action="store_true",
                        help="wait for the current solve to finish and plot before the next starts")
    parser.add_argument("--wait-seconds", type=float, default=900,
                        help="maximum wait for the active solve; the coordinator is always resumed")
    return parser.parse_args()


@contextmanager
def between_solves(input_path, enabled, timeout):
    """Stop only the coordinator, allowing its active benchmark child to finish."""
    base = str(input_path.resolve())
    if not enabled or not base.endswith("_summary.csv"):
        yield
        return
    status_path = Path(base[:-len("_summary.csv")] + "_status.json")
    if not status_path.exists():
        yield
        return
    state = json.loads(status_path.read_text())
    if state.get("phase") != "running":
        yield
        return
    worker = int(state["worker_pid"])
    try:
        os.kill(worker, 0)
    except ProcessLookupError:
        yield
        return
    stopped = False
    try:
        # A short retry covers a coordinator stopped between Popen and its
        # atomic publication of the child PID. Never signal the timed child.
        for _ in range(10):
            os.kill(worker, signal.SIGSTOP)
            stopped = True
            time.sleep(.05)
            state = json.loads(status_path.read_text())
            current = state.get("current")
            if not current or current.get("child_pid"):
                break
            os.kill(worker, signal.SIGCONT)
            stopped = False
            time.sleep(.05)
        else:
            raise RuntimeError("could not locate the campaign's active solve")
        if current:
            command = current["command"]
            result = Path(command[command.index("--output") + 1])
            started = time.monotonic()
            print(f"Waiting for {current['case']['case_id']} / {current['method']}; coordinator held, solve running",
                  flush=True)
            last_message = started
            while not result.exists():
                if time.monotonic() - started > timeout:
                    raise TimeoutError("active solve is still running; retry later or increase --wait-seconds")
                if time.monotonic() - last_message >= 30:
                    print(f"Waiting for the active solve: {time.monotonic()-started:.0f} s", flush=True)
                    last_message = time.monotonic()
                time.sleep(.5)
            # Result creation occurs after both the timed call and validation.
            time.sleep(.25)
        yield
    finally:
        if stopped:
            os.kill(worker, signal.SIGCONT)
            print("Campaign coordinator resumed", flush=True)


def positive(row, key):
    value = float(row[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {key} in {row['case_id']}")
    return value


def read_rows(path, size, threads, completed_only=False):
    # The campaign publishes this summary by atomic rename; one read gives a
    # consistent snapshot for every generated figure.
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    required = {"case_id", "degree", "size", "threads", "B_save", "B_exec", "repeat", "status"}
    if fields is None or not required.issubset(fields):
        raise ValueError("expected the CG campaign _summary.csv")
    rows = [r for r in rows if (size is None or int(r["size"]) == size)
            and (threads is None or int(r["threads"]) == threads)
            and (not completed_only or r["status"] == "complete")]
    if not rows:
        raise ValueError("no configurations match the requested filters")
    seen = set()
    for row in rows:
        key = int(row["size"]), int(row["threads"]), int(row["degree"])
        if key in seen:
            raise ValueError(f"duplicate configuration {key}")
        seen.add(key)
        if row["status"] == "complete":
            if any(int(row[f"{m}_completed"]) != int(row["repeat"]) for m in ("global", "cellwise")):
                raise ValueError(f"incomplete repetition set in {row['case_id']}")
            for method in ("global", "cellwise"):
                positive(row, f"{method}_tts_s_median")
                positive(row, f"{method}_s_per_iteration_median")
    return fields, rows


def plot(rows, metric, output_dir, time_scale, snapshot_time, plt):
    rows = sorted(rows, key=lambda r: int(r["degree"]))
    size, threads = int(rows[0]["size"]), int(rows[0]["threads"])
    fig, ax = plt.subplots(figsize=(9.0, 5.3), constrained_layout=True)
    width = .32
    measured = [r for r in rows if r["status"] == "complete"]
    values = []
    for position, row in enumerate(rows):
        if row["status"] != "complete":
            ax.axvspan(position - .4, position + .4, color="0.95", zorder=0)
            label = "failed" if row["status"] == "failed" else "pending"
            ax.text(position, .12, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="center", color="0.45", fontsize=10)
            continue
        if metric == "tts":
            g, c = [positive(row, f"{m}_tts_s_median") for m in ("global", "cellwise")]
            speedup = g / c
        elif metric == "throughput":
            dofs = (size * int(row["degree"]) + 1) ** 3
            gt, ct = [positive(row, f"{m}_s_per_iteration_median") for m in ("global", "cellwise")]
            g, c = 1e-9 * dofs / gt, 1e-9 * dofs / ct
            speedup = gt / ct
        else:
            g = .5 * (int(row["global_iterations_min"]) + int(row["global_iterations_max"]))
            c = .5 * (int(row["cellwise_iterations_min"]) + int(row["cellwise_iterations_max"]))
            speedup = None
        ax.bar(position - width / 2, g, width=width, color="tab:blue", alpha=.88)
        bars = ax.bar(position + width / 2, c, width=width, color="tab:green", alpha=.88)
        values.extend((g, c))
        if speedup is not None:
            bar = bars[0]
            ax.annotate(f"{speedup:.2f} X", xy=(bar.get_x() + bar.get_width() / 2, c),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        else:
            for method, center in (("global", position - width / 2), ("cellwise", position + width / 2)):
                lo, hi = int(row[f"{method}_iterations_min"]), int(row[f"{method}_iterations_max"])
                ax.errorbar(center, (lo + hi) / 2, yerr=(hi - lo) / 2,
                            color="0.25", capsize=4, linewidth=1)
                ax.annotate(str(lo) if lo == hi else f"{lo}–{hi}", (center, hi), xytext=(0, 5),
                            textcoords="offset points", ha="center", va="bottom", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="tab:blue", alpha=.88, label="global: gather + apply + scatter"),
                       Patch(facecolor="tab:green", alpha=.88, label="cellwise: apply + DSS (fixed B)")],
              loc="upper left", fontsize=9)
    metric_titles = {"tts": "CG time to solution", "throughput": "CG iteration throughput", "iterations": "CG iterations"}
    thread_label = "thread" if threads == 1 else "threads"
    ax.set_title(f"{metric_titles[metric]}, mesh ${size}^3$, {threads} {thread_label}\n"
                 f"{len(measured)}/{len(rows)} complete configurations; B from the apply+DSS benchmark")
    ax.set_xlabel("Polynomial order p\nB_save / B_exec (cellwise)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r['degree']}\n{r['B_save']}/{r['B_exec']}" for r in rows])
    ax.set_xlim(-.6, len(rows) - .4)
    ax.grid(axis="y", linestyle=":", linewidth=.8, alpha=.55)
    if metric == "tts":
        ax.set_yscale(time_scale)
        ax.set_ylabel("Time to solution [s]" + (" (log scale)" if time_scale == "log" else ""))
        if values:
            ax.set_ylim(min(values) * .35, max(values) * 2.8) if time_scale == "log" else ax.set_ylim(0, max(values) * 1.42)
        else:
            ax.set_ylim(.1, 10) if time_scale == "log" else ax.set_ylim(0, 1)
        note = "Median complete solve times; labels = global / cellwise time"
    elif metric == "throughput":
        ax.set_ylabel("Unique CG iteration throughput [GDoF/s]")
        ax.set_ylim(0, max(values) * 1.42 if values else 1)
        note = "N_global / median(T_CG / iterations); includes solve initialization; labels = iteration speedup"
    else:
        ax.set_ylabel("Iterations")
        ax.set_ylim(0, max(values) * 1.42 if values else 1)
        note = "Bars = midpoint of observed range; error bars and labels = minimum–maximum"
    fig.supxlabel(f"{note}\nComplete repetition sets only; snapshot {snapshot_time}", fontsize=8)
    output = output_dir / f"dss_bxyze_cg_{metric}_n{size}_t{threads}_median.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    args = arguments()
    # Import matplotlib only in the idle interval, since font/cache setup and
    # rendering can otherwise compete with an active timed benchmark.
    with between_solves(args.input, args.between_solves, args.wait_seconds):
        cache = Path(tempfile.gettempdir()) / "sfem_dss_matplotlib_cache"
        os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fields, rows = read_rows(args.input.resolve(), args.size, args.threads, args.completed_only)
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_time = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        snapshot = output_dir / "dss_bxyze_cg_plot_data.csv"
        with snapshot.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        groups = sorted({(int(r["size"]), int(r["threads"])) for r in rows})
        for size, threads in groups:
            selected = [r for r in rows if int(r["size"]) == size and int(r["threads"]) == threads]
            for metric in dict.fromkeys(args.metrics):
                print(plot(selected, metric, output_dir, args.time_scale, snapshot_time, plt), flush=True)
        print(f"Plotted {sum(r['status']=='complete' for r in rows)}/{len(rows)} completed configurations", flush=True)


if __name__ == "__main__":
    main()
