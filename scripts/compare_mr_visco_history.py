#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def round_cols(df, cols, ndigits=8):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].round(ndigits)
    return df


def summarize_diff(df, label_col, base_col, per_elem_col):
    rows = []
    for label, sub in df.groupby(label_col):
        diff = sub[per_elem_col] - sub[base_col]
        denom = sub[base_col].replace(0.0, np.nan)
        rel = diff / denom
        rows.append(
            {
                label_col: label,
                "count": len(sub),
                "max_abs_diff": float(np.nanmax(np.abs(diff))),
                "rmse": float(np.sqrt(np.nanmean(diff**2))),
                "max_rel_diff": float(np.nanmax(np.abs(rel))),
            }
        )
    return pd.DataFrame(rows)


def plot_visco_validation(df, out_dir):
    modes = df["mode"].unique()
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        sub = df[df["mode"] == mode].copy()
        sub = sub.sort_values("time")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(sub["strain"], sub["stress_sfem_base"], "b-", label="baseline")
        axes[0].plot(sub["strain"], sub["stress_sfem_per_elem"], "r--", label="per_elem")
        if "stress_marc" in sub:
            axes[0].plot(sub["strain"], sub["stress_marc"], "k:", label="Marc")
        axes[0].set_title(f"{mode} - Stress vs Strain")
        axes[0].set_xlabel("Strain")
        axes[0].set_ylabel("Stress [MPa]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        diff = sub["stress_sfem_per_elem"] - sub["stress_sfem_base"]
        axes[1].plot(sub["time"], diff, "m-")
        axes[1].set_title(f"{mode} - Stress Difference (per_elem - baseline)")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Stress diff [MPa]")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"visco_validation_diff_{mode}.png", dpi=150)
        plt.close(fig)


def plot_strain_rate(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rates = df["strain_rate"].unique()

    for rate in rates:
        sub = df[df["strain_rate"] == rate].copy()
        sub = sub.sort_values("strain")
        diff = sub["stress_MPa_per_elem"] - sub["stress_MPa_base"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(sub["strain"], sub["stress_MPa_base"], "b-", label="baseline")
        axes[0].plot(sub["strain"], sub["stress_MPa_per_elem"], "r--", label="per_elem")
        axes[0].set_title(f"Strain rate {rate} - Stress vs Strain")
        axes[0].set_xlabel("Strain")
        axes[0].set_ylabel("Stress [MPa]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(sub["strain"], diff, "m-")
        axes[1].set_title(f"Strain rate {rate} - Stress Difference")
        axes[1].set_xlabel("Strain")
        axes[1].set_ylabel("Stress diff [MPa]")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"strain_rate_diff_{rate}.png", dpi=150)
        plt.close(fig)


def compare_visco_validation(baseline_dir, per_elem_dir, out_dir):
    base_csv = baseline_dir / "visco_validation_results.csv"
    elem_csv = per_elem_dir / "visco_validation_results.csv"
    if not base_csv.exists() or not elem_csv.exists():
        return None

    base = pd.read_csv(base_csv)
    elem = pd.read_csv(elem_csv)

    base = round_cols(base, ["time", "strain"])
    elem = round_cols(elem, ["time", "strain"])

    merged = pd.merge(
        base,
        elem,
        on=["mode", "time", "strain"],
        suffixes=("_base", "_per_elem"),
    )

    summary = summarize_diff(merged, "mode", "stress_sfem_base", "stress_sfem_per_elem")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "visco_validation_merged.csv", index=False)
    summary.to_csv(out_dir / "visco_validation_diff_summary.csv", index=False)

    plot_visco_validation(merged, out_dir)
    return summary


def find_displacement_files(output_dir):
    for extension in ("float64", "raw"):
        component_files = []
        for comp in range(3):
            files = []
            for path in output_dir.glob(f"disp.{comp}.*.{extension}"):
                try:
                    files.append((int(path.name.split(".")[2]), path))
                except (IndexError, ValueError):
                    continue
            component_files.append(sorted(files))

        if any(component_files):
            counts = [len(files) for files in component_files]
            if not all(counts) or any(count != counts[0] for count in counts[1:]):
                raise RuntimeError(f"displacement component frame counts differ in {output_dir}: {counts}")
            return extension, component_files

    raise FileNotFoundError(f"no displacement files found in {output_dir}")


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
    components = [np.fromfile(files[frame][1], dtype=np.float64) for files in component_files]
    if any(comp.shape != components[0].shape for comp in components[1:]):
        raise RuntimeError(f"displacement component sizes differ at frame {frame}")
    return components


def plot_spatial_error(diffs, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    for diff, color, label in zip(diffs, ("blue", "green", "red"), ("X", "Y", "Z")):
        ax.plot(np.abs(diff), color=color, linewidth=0.8, label=f"|diff| {label}")
    ax.set_title("Spatial: Final Step Node-wise Error Distribution", fontweight="bold")
    ax.set_xlabel("Node Index")
    ax.set_ylabel("|diff|")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spatial_error_distribution.png", dpi=150)
    plt.close(fig)


def plot_temporal_error(temporal, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(temporal["time"], temporal["max_abs_diff"], "r-o", markersize=2)
    axes[0].set_title("Maximum Absolute Difference Over Time")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("max |diff|")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(temporal["time"], temporal["relative_l2_percent"], "g-o", markersize=2)
    axes[1].set_title("Relative Error Over Time (||diff|| / ||baseline||)")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Relative L2 Error (%)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Error Summary: per_elem vs baseline", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "error_summary_simple.png", dpi=150)
    plt.close(fig)


def compare_gravity(baseline_dir, per_elem_dir, out_dir):
    base_out = baseline_dir / "test_mooney_rivlin_gravity"
    elem_out = per_elem_dir / "test_mooney_rivlin_gravity"
    if not base_out.exists() or not elem_out.exists():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    base_extension, base_files = find_displacement_files(base_out)
    elem_extension, elem_files = find_displacement_files(elem_out)
    base_step_ids = [[step for step, _ in files] for files in base_files]
    elem_step_ids = [[step for step, _ in files] for files in elem_files]
    if base_step_ids != elem_step_ids:
        raise RuntimeError("baseline and per-element displacement field IDs are not aligned")

    n_frames = len(base_files[0])
    base_times = read_times(base_out, n_frames)
    elem_times = read_times(elem_out, n_frames)
    if not np.allclose(base_times, elem_times, rtol=0.0, atol=1e-14):
        raise RuntimeError("baseline and per-element physical times are not aligned")

    temporal_rows = []
    final_diffs = None
    for frame, time in enumerate(base_times):
        base_components = load_displacement(base_files, frame)
        elem_components = load_displacement(elem_files, frame)
        if any(base.shape != elem.shape for base, elem in zip(base_components, elem_components)):
            raise RuntimeError(f"baseline and per-element displacement sizes differ at frame {frame}")

        diffs = [elem - base for base, elem in zip(base_components, elem_components)]
        diff_all = np.concatenate(diffs)
        base_all = np.concatenate(base_components)
        baseline_norm = np.linalg.norm(base_all)
        temporal_rows.append(
            {
                "time": float(time),
                "disp0_output_field": base_step_ids[0][frame],
                "max_abs_diff": float(np.max(np.abs(diff_all))),
                "rmse": float(np.sqrt(np.mean(diff_all**2))),
                "relative_l2_percent": float(np.linalg.norm(diff_all) / baseline_norm * 100)
                if baseline_norm > 0
                else 0.0,
            }
        )
        final_diffs = diffs

    temporal = pd.DataFrame(temporal_rows)
    temporal.to_csv(out_dir / "temporal_error_analysis.csv", index=False)

    comp_summaries = []
    for comp, diff in enumerate(final_diffs):
        comp_summary = {
            "component": comp,
            "n_nodes": diff.shape[0],
            "output_field": base_step_ids[comp][-1],
            "time": float(base_times[-1]),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff**2))),
        }
        comp_summaries.append(comp_summary)
        pd.DataFrame([comp_summary]).to_csv(out_dir / f"gravity_diff_comp{comp}.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.abs(diff), "m-")
        ax.set_title(f"Gravity disp diff comp {comp} (per_elem - baseline)")
        ax.set_xlabel("Node index")
        ax.set_ylabel("Abs diff")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"gravity_diff_nodes_comp{comp}.png", dpi=150)
        plt.close(fig)

    final_diff_all = np.concatenate(final_diffs)
    peak_max_idx = temporal["max_abs_diff"].idxmax()
    peak_relative_idx = temporal["relative_l2_percent"].idxmax()
    summary = {
        "n_nodes": final_diffs[0].shape[0],
        "n_frames": len(temporal),
        "final_time": float(base_times[-1]),
        "baseline_format": base_extension,
        "per_elem_format": elem_extension,
        "max_abs_diff": float(np.max(np.abs(final_diff_all))),
        "rmse": float(np.sqrt(np.mean(final_diff_all**2))),
        "relative_l2_percent": float(temporal.iloc[-1]["relative_l2_percent"]),
        "peak_max_abs_diff": float(temporal.loc[peak_max_idx, "max_abs_diff"]),
        "peak_max_abs_time": float(temporal.loc[peak_max_idx, "time"]),
        "peak_relative_l2_percent": float(temporal.loc[peak_relative_idx, "relative_l2_percent"]),
        "peak_relative_l2_time": float(temporal.loc[peak_relative_idx, "time"]),
    }
    pd.DataFrame(comp_summaries).to_csv(out_dir / "gravity_diff_summary_by_comp.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "gravity_diff_summary.csv", index=False)
    plot_spatial_error(final_diffs, out_dir)
    plot_temporal_error(temporal, out_dir)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--per-elem", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    per_elem_dir = Path(args.per_elem)
    out_dir = Path(args.out)

    script_root = Path(__file__).resolve().parent.parent
    if not baseline_dir.exists():
        alt = script_root / baseline_dir
        if alt.exists():
            baseline_dir = alt
    if not per_elem_dir.exists():
        alt = script_root / per_elem_dir
        if alt.exists():
            per_elem_dir = alt
    if not out_dir.exists():
        alt = script_root / out_dir
        if alt.exists():
            out_dir = alt

    out_dir.mkdir(parents=True, exist_ok=True)

    visco_summary = compare_visco_validation(baseline_dir, per_elem_dir, out_dir)
    gravity_summary = compare_gravity(baseline_dir, per_elem_dir, out_dir)

    summary_txt = out_dir / "summary.txt"
    with summary_txt.open("w") as f:
        f.write("MooneyRivlinVisco history compare summary\n")
        f.write("========================================\n\n")
        if visco_summary is not None:
            f.write("Visco Excel Validation:\n")
            visco_print = visco_summary.copy()
            for col in ["max_abs_diff", "rmse", "max_rel_diff"]:
                if col in visco_print.columns:
                    visco_print[col] = visco_print[col].astype(float)
            f.write(visco_print.to_string(index=False, float_format="%.8f"))
            f.write("\n\n")
        else:
            f.write("Visco Excel Validation: missing CSVs\n\n")
        if gravity_summary is not None:
            f.write("Gravity Test (final displacement):\n")
            for k, v in gravity_summary.items():
                if isinstance(v, float):
                    f.write(f"{k}: {v:.8e}\n")
                else:
                    f.write(f"{k}: {v}\n")
            f.write("\n")
        else:
            f.write("Gravity Test: missing outputs\n\n")


if __name__ == "__main__":
    main()
