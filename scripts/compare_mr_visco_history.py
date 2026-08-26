#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_displacement_files(output_dir):
    component_files = [
        sorted(output_dir.glob(f"disp.{comp}.*.float64"), key=lambda path: int(path.name.split(".")[2]))
        for comp in range(3)
    ]
    counts = [len(files) for files in component_files]
    if not all(counts) or len(set(counts)) != 1:
        raise RuntimeError(f"expected aligned FP64 displacement components in {output_dir}, got {counts} frames")
    return component_files


def read_times(output_dir, expected_count):
    path = output_dir / "time.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing physical time file: {path}")

    times = np.atleast_1d(np.loadtxt(path, dtype=np.float64))
    if len(times) != expected_count:
        raise RuntimeError(
            f"{path} contains {len(times)} times but there are {expected_count} displacement frames; "
            "the output directory may contain stale files"
        )
    return times


def load_displacement(component_files, frame):
    return [np.fromfile(files[frame], dtype=np.float64) for files in component_files]


def plot_spatial_error(diffs, out_dir, comparison_label):
    fig, ax = plt.subplots(figsize=(12, 5))
    for diff, color, label in zip(diffs, ("blue", "green", "red"), ("X", "Y", "Z")):
        ax.plot(np.abs(diff), color=color, linewidth=0.8, label=f"|diff| {label}")
    ax.set_title(f"Final-step Node-wise Error: {comparison_label}", fontweight="bold")
    ax.set_xlabel("Node Index")
    ax.set_ylabel("|diff|")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spatial_error_distribution.png", dpi=150)
    plt.close(fig)


def plot_temporal_error(temporal, out_dir, comparison_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(temporal["time"], temporal["max_abs_diff"], "r-o", markersize=2)
    axes[0].set_title("Maximum Absolute Difference Over Time")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("max |diff|")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(temporal["time"], temporal["relative_l2_percent"], "g-o", markersize=2)
    axes[1].set_title("Relative Error Over Time (||diff|| / ||reference||)")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Relative L2 Error (%)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Error Summary: {comparison_label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "error_summary_simple.png", dpi=150)
    plt.close(fig)


def compare_displacement(reference_dir, candidate_dir, out_dir, output_subdir, reference_label, candidate_label):
    reference_out = reference_dir / output_subdir
    candidate_out = candidate_dir / output_subdir
    comparison_label = f"{candidate_label} - {reference_label}"

    reference_files = find_displacement_files(reference_out)
    candidate_files = find_displacement_files(candidate_out)
    n_frames = len(reference_files[0])
    if len(candidate_files[0]) != n_frames:
        raise RuntimeError("reference and candidate frame counts differ")

    reference_times = read_times(reference_out, n_frames)
    candidate_times = read_times(candidate_out, n_frames)
    if not np.allclose(reference_times, candidate_times, rtol=0.0, atol=1e-14):
        raise RuntimeError("reference and candidate physical times are not aligned")

    temporal_rows = []
    final_diffs = None
    for frame, time in enumerate(reference_times):
        reference_components = load_displacement(reference_files, frame)
        candidate_components = load_displacement(candidate_files, frame)
        if len({component.size for component in reference_components + candidate_components}) != 1:
            raise RuntimeError(f"reference and candidate displacement sizes differ at frame {frame}")

        diffs = [cand - ref for ref, cand in zip(reference_components, candidate_components)]
        diff_all = np.concatenate(diffs)
        reference_all = np.concatenate(reference_components)
        reference_norm = np.linalg.norm(reference_all)
        temporal_rows.append(
            {
                "time": float(time),
                "max_abs_diff": float(np.max(np.abs(diff_all))),
                "rmse": float(np.sqrt(np.mean(diff_all**2))),
                "relative_l2_percent": float(np.linalg.norm(diff_all) / reference_norm * 100)
                if reference_norm > 0
                else 0.0,
            }
        )
        final_diffs = diffs

    out_dir.mkdir(parents=True, exist_ok=True)
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_csv(out_dir / "temporal_error_analysis.csv", index=False)

    final_diff_all = np.concatenate(final_diffs)
    peak_max_idx = temporal["max_abs_diff"].idxmax()
    peak_relative_idx = temporal["relative_l2_percent"].idxmax()
    summary = {
        "reference": reference_label,
        "candidate": candidate_label,
        "n_nodes": final_diffs[0].shape[0],
        "n_frames": len(temporal),
        "final_time": float(reference_times[-1]),
        "final_max_abs_diff": float(np.max(np.abs(final_diff_all))),
        "final_rmse": float(np.sqrt(np.mean(final_diff_all**2))),
        "final_relative_l2_percent": float(temporal.iloc[-1]["relative_l2_percent"]),
        "peak_max_abs_diff": float(temporal.loc[peak_max_idx, "max_abs_diff"]),
        "peak_max_abs_time": float(temporal.loc[peak_max_idx, "time"]),
        "peak_relative_l2_percent": float(temporal.loc[peak_relative_idx, "relative_l2_percent"]),
        "peak_relative_l2_time": float(temporal.loc[peak_relative_idx, "time"]),
    }

    pd.DataFrame([summary]).to_csv(out_dir / "displacement_diff_summary.csv", index=False)
    plot_spatial_error(final_diffs, out_dir, comparison_label)
    plot_temporal_error(temporal, out_dir, comparison_label)

    with (out_dir / "summary.txt").open("w") as f:
        f.write(f"MooneyRivlinVisco comparison: {comparison_label}\n")
        f.write("========================================\n\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.8e}\n")
            else:
                f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare two Mooney-Rivlin viscoelastic displacement histories")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-subdir", default="test_mooney_rivlin_gravity")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    compare_displacement(
        Path(args.reference),
        Path(args.candidate),
        Path(args.out),
        args.output_subdir,
        args.reference_label,
        args.candidate_label,
    )


if __name__ == "__main__":
    main()
