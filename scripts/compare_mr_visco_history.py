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


def find_last_step_for_comp(output_dir, comp):
    files = sorted(output_dir.glob(f"disp.{comp}.*.raw"))
    if not files:
        return None
    best = None
    best_step = -1
    for p in files:
        parts = p.name.split(".")
        if len(parts) >= 4:
            try:
                step = int(parts[2])
            except Exception:
                step = -1
            if step > best_step:
                best_step = step
                best = parts[2]
    if best is None:
        best = files[-1].name.split(".")[2]
    return best


def load_comp(output_dir, comp, step_str):
    path = output_dir / f"disp.{comp}.{step_str}.raw"
    if not path.exists():
        return None
    return np.fromfile(path, dtype=np.float64)


def compare_gravity(baseline_dir, per_elem_dir, out_dir):
    base_out = baseline_dir / "test_mooney_rivlin_gravity"
    elem_out = per_elem_dir / "test_mooney_rivlin_gravity"
    if not base_out.exists() or not elem_out.exists():
        return None

    diffs = []
    comp_summaries = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for comp in range(3):
        base_step = find_last_step_for_comp(base_out, comp)
        elem_step = find_last_step_for_comp(elem_out, comp)
        if base_step is None or elem_step is None:
            return None

        base_comp = load_comp(base_out, comp, base_step)
        elem_comp = load_comp(elem_out, comp, elem_step)
        if base_comp is None or elem_comp is None:
            return None
        if base_comp.shape != elem_comp.shape:
            return None

        diff = elem_comp - base_comp
        diffs.append(diff)

        comp_summary = {
            "component": comp,
            "n_nodes": base_comp.shape[0],
            "base_step": base_step,
            "per_elem_step": elem_step,
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

    diff_all = np.concatenate(diffs)
    summary = {
        "n_nodes": diffs[0].shape[0],
        "max_abs_diff": float(np.max(np.abs(diff_all))),
        "rmse": float(np.sqrt(np.mean(diff_all**2))),
    }
    pd.DataFrame(comp_summaries).to_csv(out_dir / "gravity_diff_summary_by_comp.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "gravity_diff_summary.csv", index=False)
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
                    f.write(f"{k}: {v:.8f}\n")
                else:
                    f.write(f"{k}: {v}\n")
            f.write("\n")
        else:
            f.write("Gravity Test: missing outputs\n\n")


if __name__ == "__main__":
    main()
