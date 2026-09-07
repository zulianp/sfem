#!/usr/bin/env python3
"""Plot history errors already measured by the fixed-trajectory replay test."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_cases(run_dirs, end_time=None):
    cases = {}
    reference_times = None
    for run_dir in run_dirs:
        paths = sorted(Path(run_dir).glob("*/test_mooney_rivlin_gravity/history_replay.csv"))
        if not paths:
            raise ValueError(f"No replay CSV files in {run_dir}")
        for path in paths:
            label = path.parent.parent.name
            if label in cases:
                raise ValueError(f"Duplicate case {label}: select only one completed run per case")
            data = pd.read_csv(path)
            required = ["time", "max_abs_error", "l2_error", "relative_l2_error"]
            if not set(required).issubset(data.columns) or len(data) < 2:
                raise ValueError(f"Missing replay columns or time steps: {path}")
            if not np.isfinite(data.to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite replay data: {path}")
            times = data["time"].to_numpy()
            if abs(times[0]) > 1e-12 or np.any(np.diff(times) <= 0):
                raise ValueError(f"Expected increasing times starting at zero: {path}")
            if (data[required[1:]] < 0).any().any():
                raise ValueError(f"Negative error norm: {path}")
            # The solver's while(t < T) loop can overshoot T by one step.
            if end_time is not None and not (
                end_time - 1e-10 <= times[-1] <= end_time + np.max(np.diff(times)) + 1e-10
            ):
                raise ValueError(f"{path}: final time {times[-1]:g} does not match requested T={end_time:g}")
            if reference_times is not None and (
                times.shape != reference_times.shape
                or not np.allclose(times, reference_times, rtol=0, atol=1e-12)
            ):
                raise ValueError(f"Time grids differ: {path}")
            reference_times = times
            cases[label] = (path, data)
    if not cases:
        raise ValueError("No replay cases supplied")
    return cases


def plot_cases(cases, out_dir, log_scale=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rows = []
    for label, (path, data) in cases.items():
        absolute = data["max_abs_error"].to_numpy()
        relative = 100 * data["relative_l2_error"].to_numpy()
        for ax, values in zip(axes, (absolute, relative)):
            curve_label = label + (" (all zero)" if np.all(values == 0) else "")
            # Zero has no logarithm: leave gaps, rather than inventing an error floor.
            plotted = np.where(values > 0, values, np.nan) if log_scale else values
            ax.plot(data["time"], plotted, label=curve_label, linewidth=1.4)
        rows.append({
            "case": label,
            "source_csv": str(path.resolve()),
            "samples": len(data),
            "final_time": data["time"].iloc[-1],
            "final_max_abs_error": absolute[-1],
            "peak_max_abs_error": absolute.max(),
            "peak_max_abs_time": data["time"].iloc[int(np.argmax(absolute))],
            "final_relative_l2_percent": relative[-1],
            "peak_relative_l2_percent": relative.max(),
            "peak_relative_l2_time": data["time"].iloc[int(np.argmax(relative))],
        })
    for ax, title, ylabel in zip(
        axes,
        ("Maximum absolute history error", "Relative history error"),
        (r"$\max |\widehat{H}-H_{\mathrm{ref}}|$",
         r"$100\|\widehat{H}-H_{\mathrm{ref}}\|_2/\|H_{\mathrm{ref}}\|_2$ (%)"),
    ):
        if log_scale and any(np.any(line.get_ydata() > 0) for line in ax.lines):
            ax.set_yscale("log")
        ax.set(title=title, xlabel="Time [s]", ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Fixed-trajectory history replay: candidate vs reference")
    if log_scale:
        fig.text(0.5, 0.01, "Zero values are omitted on logarithmic axes; exact values are retained in the CSV.",
                 ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04 if log_scale else 0, 1, 0.95))
    stem = "replay_history_error_log" if log_scale else "replay_history_error"
    for extension in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{extension}", dpi=180)
    plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "replay_summary.csv", index=False)
    print(summary.drop(columns="source_csv").to_string(index=False))
    print(f"Plots and summary: {out_dir.resolve()}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Replay job directories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--end-time", type=float, help="Check that each CSV reached the requested physical time")
    parser.add_argument("--log", action="store_true", help="Use log error axes; omit zero values")
    args = parser.parse_args()
    plot_cases(load_cases(args.runs, args.end_time), args.out, args.log)


if __name__ == "__main__":
    main()
