#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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


def compare_displacement(reference_dir, candidate_dir, out_dir, output_subdir, reference_label, candidate_label,
                         tip_mask=None, end_time=None):
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
    if not np.isfinite(reference_times).all() or np.any(np.diff(reference_times) <= 0):
        raise RuntimeError("expected finite, increasing output times")
    if end_time is not None:
        # Periodic field output need not include the last solver step.
        if n_frames < 2 or abs(reference_times[-1] - end_time) > np.max(np.diff(reference_times)) + 1e-10:
            raise RuntimeError(f"last output time {reference_times[-1]:g} does not reach T={end_time:g}")

    temporal_rows = []
    final_diffs = None
    for frame, time in enumerate(reference_times):
        reference_components = load_displacement(reference_files, frame)
        candidate_components = load_displacement(candidate_files, frame)
        if len({component.size for component in reference_components + candidate_components}) != 1:
            raise RuntimeError(f"reference and candidate displacement sizes differ at frame {frame}")
        if not all(np.isfinite(component).all() for component in reference_components + candidate_components):
            raise RuntimeError(f"non-finite displacement at frame {frame}")

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
        if tip_mask is not None:
            if tip_mask.size != reference_components[0].size:
                raise RuntimeError("mesh coordinates and displacement node counts differ")
            temporal_rows[-1].update(
                reference_tip_uy=float(reference_components[1][tip_mask].mean()),
                candidate_tip_uy=float(candidate_components[1][tip_mask].mean()),
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
    return temporal


def plot_comparisons(comparisons, out_dir, reference_label):
    has_tip = "reference_tip_uy" in next(iter(comparisons.values()))
    fig, axes = plt.subplots(1, 3 if has_tip else 2, figsize=(17 if has_tip else 12, 5))
    if has_tip:
        first = next(iter(comparisons.values()))
        axes[0].plot(first["time"], first["reference_tip_uy"], color="black", label=reference_label)
        axes[0].set(title="Free-end section mean displacement", ylabel="Mean $u_y$ (length units)")
    error_axes = axes[-2:]
    for i, (label, data) in enumerate(comparisons.items()):
        style = ("-", "--", "-.", ":")[i % 4]
        if has_tip:
            axes[0].plot(data["time"], data["candidate_tip_uy"], linestyle=style, label=label)
        for ax, column in zip(error_axes, ("max_abs_diff", "relative_l2_percent")):
            values = data[column].to_numpy()
            ax.plot(data["time"], np.where(values > 0, values, np.nan), linestyle=style,
                    marker=("o", "s", "^", "x")[i % 4], markersize=3,
                    markevery=max(1, len(values) // 15),
                    label=label + (" (all zero)" if np.all(values == 0) else ""))
    for ax, title, ylabel in zip(error_axes,
                                ("Whole-field maximum component error", "Whole-field relative displacement error"),
                                ("max |mixed - FP64|", "Relative Euclidean error (%)")):
        if any(np.any(line.get_ydata() > 0) for line in ax.lines):
            ax.set_yscale("log")
        ax.set(title=title, ylabel=ylabel)
    for ax in axes:
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Fully coupled displacement comparison")
    fig.text(0.5, 0.015, "Error axes omit zeros; exact values are retained in the CSV.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    for extension in ("png", "pdf"):
        fig.savefig(out_dir / f"coupled_comparison.{extension}", dpi=180)
    plt.close(fig)
    pd.concat([data.assign(candidate=label) for label, data in comparisons.items()],
              ignore_index=True).to_csv(out_dir / "coupled_comparison.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Compare coupled displacement histories against one reference")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True, action="append", help="Repeat for multiple candidates")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", action="append", help="One label per candidate; defaults to directory names")
    parser.add_argument("--mesh", type=Path, help="Shared box_mesh.py mesh (x.raw is float32); tip is x=max(x)")
    parser.add_argument("--end-time", type=float, help="Check output reached T within one export interval")
    parser.add_argument("--output-subdir", default="test_mooney_rivlin_gravity")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    labels = args.candidate_label or [Path(path).name for path in args.candidate]
    if len(labels) != len(args.candidate) or len(set(labels)) != len(labels):
        parser.error("provide one unique label per candidate")
    tip_mask = None
    if args.mesh:
        x = np.fromfile(args.mesh / "x.raw", dtype=np.float32)
        if not x.size or not np.isfinite(x).all():
            parser.error("empty or non-finite mesh coordinates")
        # box_mesh.py generates an exactly planar x=max(x) end face.
        tip_mask = x == x.max()
        print(f"Free end: x={x.max():g}, {tip_mask.sum()} nodes; arithmetic mean of nodal uy")
    out_dir = Path(args.out)
    comparisons = {}
    for candidate, label in zip(args.candidate, labels):
        pair_out = out_dir / Path(candidate).name if len(labels) > 1 else out_dir
        comparisons[label] = compare_displacement(
            Path(args.reference), Path(candidate), pair_out, args.output_subdir,
            args.reference_label, label, tip_mask, args.end_time,
        )
    if len(labels) > 1 or tip_mask is not None:
        plot_comparisons(comparisons, out_dir, args.reference_label)


if __name__ == "__main__":
    main()
