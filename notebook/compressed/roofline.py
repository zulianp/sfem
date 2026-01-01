#!/usr/bin/env python3
"""
Roofline plot (polished, publication-ready defaults).

This script is meant to be a faithful companion to `LAPLACIAN.md`: it encodes the
same FLOP/byte assumptions in a structured form, prints a compact table, and
generates a roofline plot.

Units:
- peak_gflops:   GFLOP/s
- mem_bw_gbs:    GB/s  (sustained)
- op_intensity:  FLOP/byte
- achieved_gflops: GFLOP/s
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class KernelPoint:
    name: str
    op_intensity: float      # FLOP / byte
    achieved_gflops: Optional[float]   # GFLOP / s (None => theoretical-only)
    note: str = ""           # optional label details


@dataclass(frozen=True)
class AnalysisCase:
    name: str
    flops_per_elem: float
    bytes_per_elem: float
    note: str = ""

    def op_intensity(self) -> float:
        return self.flops_per_elem / self.bytes_per_elem


def roofline_gflops(oi, peak_gflops: float, mem_bw_gbs: float):
    """
    Roofline model:
      P(oi) = min(peak, BW * oi)
    where BW in GB/s and oi in flop/byte -> BW*oi is GFLOP/s.
    """
    # Works for scalars and numpy arrays (if numpy is available).
    try:
        import numpy as np  # type: ignore

        return np.minimum(peak_gflops, mem_bw_gbs * oi)
    except ModuleNotFoundError:
        if isinstance(oi, (list, tuple)):
            return [min(peak_gflops, mem_bw_gbs * x) for x in oi]
        return min(peak_gflops, mem_bw_gbs * oi)


def nice_log_limits(vals: List[float], pad: float = 4.0) -> tuple[float, float]:
    vmin = min(vals)
    vmax = max(vals)
    vmin = max(vmin / pad, 1e-6)
    vmax = vmax * pad
    return vmin, vmax


def format_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def build_laplacian_cases() -> List[AnalysisCase]:
    # NOTE: these match the byte accounting in notebook/compressed/LAPLACIAN.md.
    # FLOPs/elem: (340 + 8)
    flops = 340.0 + 8.0

    return [
        AnalysisCase(
            name="New (fp64)",
            flops_per_elem=flops,
            bytes_per_elem=6 * 4 + 8 * 8 + 8 * 8 + 8 * 2,
            note="",
        ),
        AnalysisCase(
            name="Original (fp64)",
            flops_per_elem=flops,
            bytes_per_elem=6 * 4 + 8 * 8 + 8 * 8 + 8 * 8,
            note="",
        ),
        # AnalysisCase(
        #     name="vector-packed (3 elems, fp64)",
        #     flops_per_elem=3.0 * flops,
        #     bytes_per_elem=6 * 4 + 3 * 8 * 8 + 3 * 8 * 8 + 8 * 2,
        #     note="3 elements share fff+idx; u/out fp64",
        # ),
        # AnalysisCase(
        #     name="vector-packed (3 elems, fp32)",
        #     flops_per_elem=3.0 * flops,
        #     bytes_per_elem=6 * 4 + 3 * 8 * 4 + 3 * 8 * 4 + 8 * 2,
        #     note="3 elements share fff+idx; u/out fp32",
        # ),
    ]


def achieved_gflops(flops_per_elem: float, n_elems: float, time_s: float) -> float:
    return 1e-9 * flops_per_elem * n_elems / time_s


def compute_points(
    cases: List[AnalysisCase],
    achieved: float,
    achieved_case: str,
) -> List[KernelPoint]:
    return [
        KernelPoint(
            name=c.name,
            op_intensity=c.op_intensity(),
            achieved_gflops=(achieved if c.name == achieved_case else None),
            note=c.note,
        )
        for c in cases
    ]


def print_summary_table(
    cases: List[AnalysisCase],
    achieved: float,
    achieved_case: str,
    peak_gflops: float,
    mem_bw_gbs: float,
) -> None:
    ridge_oi = peak_gflops / mem_bw_gbs
    print(f"Peak: {peak_gflops:.2f} GFLOP/s, BW: {mem_bw_gbs:.2f} GB/s, ridge OI: {ridge_oi:.3f} FLOP/B")
    print()
    header = ("case", "FLOP/elem", "B/elem", "OI [F/B]", "roof [GF/s]", "ach [GF/s]", "%roof", "%peak")

    rows: List[Tuple[str, ...]] = []
    for c in cases:
        oi = c.op_intensity()
        roof = min(peak_gflops, mem_bw_gbs * oi)
        is_measured = (c.name == achieved_case)
        ach_str = f"{achieved:.1f}" if is_measured else "-"
        pct_roof_str = format_pct(achieved / roof) if is_measured else "-"
        pct_peak_str = format_pct(achieved / peak_gflops) if is_measured else "-"
        rows.append(
            (
                c.name,
                f"{c.flops_per_elem:.0f}",
                f"{c.bytes_per_elem:.0f}",
                f"{oi:.4f}",
                f"{roof:.1f}",
                ach_str,
                pct_roof_str,
                pct_peak_str,
            )
        )

    # Compute widths from actual data (keeps it readable even with long case names).
    widths = [len(h) for h in header]
    for r in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, r)]

    # Left-align case name, right-align numeric columns.
    def fmt_row(cols: Tuple[str, ...]) -> str:
        out = []
        for i, (cell, w) in enumerate(zip(cols, widths)):
            if i == 0:
                out.append(cell.ljust(w))
            else:
                out.append(cell.rjust(w))
        return "  ".join(out)

    print(fmt_row(header))
    print(fmt_row(tuple("-" * w for w in widths)))
    for r in rows:
        print(fmt_row(r))
    print()


def plot_roofline(
    points: List[KernelPoint],
    peak_gflops: float,
    mem_bw_gbs: float,
    title: str = "Roofline (GFLOP/s vs FLOP/byte)",
    out_png: Optional[Path] = Path("roofline.png"),
    out_pdf: Optional[Path] = Path("roofline.pdf"),
) -> None:
    # Plotting is optional: keep the numeric summary usable even without deps.
    try:
        import numpy as np  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:
        print(f"[roofline.py] Plotting skipped: missing optional dependency: {e.name}")
        print("[roofline.py] Install numpy + matplotlib to generate plots.")
        return

    if peak_gflops <= 0 or mem_bw_gbs <= 0:
        raise ValueError("peak_gflops and mem_bw_gbs must be > 0.")
    if not points:
        raise ValueError("Need at least one KernelPoint.")

    # Style: clean, readable defaults (no extra deps beyond matplotlib).
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    # Ridge point (operational intensity at the intersection)
    ridge_oi = peak_gflops / mem_bw_gbs

    # OI range for the roofline curve
    oi_vals = [p.op_intensity for p in points] + [ridge_oi]
    oi_min, oi_max = nice_log_limits(oi_vals, pad=6.0)

    oi_curve = np.logspace(math.log10(oi_min), math.log10(oi_max), 400)
    roof = roofline_gflops(oi_curve, peak_gflops, mem_bw_gbs)

    # Y range
    perf_vals = [p.achieved_gflops for p in points if p.achieved_gflops is not None] + [peak_gflops, mem_bw_gbs * oi_min]
    y_min, y_max = nice_log_limits(perf_vals, pad=6.0)

    # 16:9 aspect ratio (presentation-friendly), smaller footprint
    fig, ax = plt.subplots(figsize=(9.6 * 0.75, 5.4 * 0.75), dpi=170, constrained_layout=True)

    # Roofline curve and compute roof
    ax.loglog(oi_curve, roof, linewidth=2.2, color="black", label="Roofline: min(Peak, BW·OI)")
    ax.axhline(peak_gflops,
               linewidth=1.4,
               linestyle="--",
               color="black",
               alpha=0.75,
               label=f"Compute peak = {peak_gflops:.0f} GFLOP/s")

    # Ridge point marker
    ax.axvline(ridge_oi,
               linewidth=1.0,
               linestyle=":",
               color="black",
               alpha=0.65,
               label=f"Ridge OI = {ridge_oi:.2f} FLOP/B")

    # Subtle region shading (helps readability)
    ax.fill_between(oi_curve, y_min, roof, where=oi_curve <= ridge_oi, alpha=0.06, color="tab:blue")
    ax.fill_between(oi_curve, y_min, roof, where=oi_curve >= ridge_oi, alpha=0.06, color="tab:orange")

    # Plot kernel points
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "D", "^", "v", "P", "X", "h"]

    for i, p in enumerate(points):
        color = cmap(i % 10)
        marker = markers[i % len(markers)]
        roof_at_p = min(peak_gflops, mem_bw_gbs * p.op_intensity)

        # Theoretical bound at this OI (on the roofline): hollow marker
        ax.loglog([p.op_intensity],
                  [roof_at_p],
                  marker=marker,
                  markersize=8,
                  linestyle="None",
                  markerfacecolor="none",
                  markeredgecolor=color,
                  markeredgewidth=1.6,
                  alpha=0.97)

        if p.achieved_gflops is not None:
            # Achieved performance: filled marker
            ax.loglog([p.op_intensity],
                      [p.achieved_gflops],
                      marker=marker,
                      markersize=8,
                      linestyle="None",
                      markerfacecolor=color,
                      markeredgecolor="black",
                      markeredgewidth=0.6,
                      label=p.name)

            # Connector (shows headroom)
            ax.plot([p.op_intensity, p.op_intensity],
                    [p.achieved_gflops, roof_at_p],
                    linewidth=1.0,
                    color=color,
                    alpha=0.55)

            pct_roof = 100.0 * p.achieved_gflops / roof_at_p
            label = f"{p.name}\nmeasured: {p.achieved_gflops:.0f} [GFLOP/s]\nbound: {roof_at_p:.0f} [GFLOP/s]\n{pct_roof:.0f}% roof"
            ax.annotate(
                label,
                (p.op_intensity, p.achieved_gflops),
                textcoords="offset points",
                xytext=(12, -24),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.20", alpha=0.12, linewidth=0.6),
            )
        else:
            # Theoretical-only: keep it uncluttered.
            ax.annotate(
                f"{p.name}\nbound: {roof_at_p:.0f} [GFLOP/s]",
                (p.op_intensity, roof_at_p),
                textcoords="offset points",
                xytext=(-80, 10),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.18", alpha=0.08, linewidth=0.6),
            )

    # Styling
    ax.set_title(title)
    ax.set_xlabel("Operational intensity [FLOP / byte]")
    ax.set_ylabel("Performance [GFLOP / s]")
    ax.set_xlim(oi_min, oi_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)

    # Legend: put it outside to avoid covering data.
    # Also add explicit entries for achieved vs theoretical bound markers.
    from matplotlib.lines import Line2D  # type: ignore

    proxy_ach = Line2D([0], [0], marker="o", linestyle="None",
                       markerfacecolor="gray", markeredgecolor="black",
                       markersize=8, label="Achieved")
    proxy_roof = Line2D([0], [0], marker="o", linestyle="None",
                        markerfacecolor="none", markeredgecolor="gray",
                        markeredgewidth=1.6, markersize=8,
                        label="Theoretical bound (OI)")

    handles, labels = ax.get_legend_handles_labels()
    handles = [proxy_ach, proxy_roof] + handles
    labels = ["Achieved", "Theoretical bound @ OI"] + labels
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)

    # Add a small info box (numbers used)
    info = f"BW = {mem_bw_gbs:.0f} [GB/s]\nPeak = {peak_gflops:.0f} [GFLOP/s]"
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
    )

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        print(f"Wrote {out_png}")
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Wrote {out_pdf}")

    plt.close(fig)


def main() -> None:
    #  Defaults from "Accelerating Scientific Workflows with the NVIDIA Grace Hopper Platform"
    parser = argparse.ArgumentParser(description="Laplace operator on one NVIDIA Grace CPU (72 cores)")
    parser.add_argument("--peak-gflops", type=float, default=3600.0, help="Measured compute peak [GFLOP/s]")
    parser.add_argument("--mem-bw-gbs", type=float, default=500.0, help="Measured sustained memory BW [GB/s]")
    parser.add_argument("--n-elems", type=float, default=64_000_000.0, help="Number of elements.")
    parser.add_argument("--time-s", type=float, default=2.689e-2, help="Measured runtime [s]")
    parser.add_argument(
        "--achieved-case",
        type=str,
        default="New (fp64)",
        help="Case name that the provided --n-elems/--time-s corresponds to.",
    )
    parser.add_argument("--out-png", type=str, default="roofline_laplacian.png", help="Output PNG file.")
    parser.add_argument("--out-pdf", type=str, default="roofline_laplacian.pdf", help="Output PDF file.")
    args = parser.parse_args()

    peak_gflops = args.peak_gflops
    mem_bw_gbs = args.mem_bw_gbs

    cases = build_laplacian_cases()

    # Achieved performance is taken from the measured runtime for the baseline kernel.
    # (We plot the same achieved point for each case, so the chart directly shows how
    # close you are to each variant's roof.)
    achieved = achieved_gflops(flops_per_elem=(340.0 + 8.0), n_elems=args.n_elems, time_s=args.time_s)

    print_summary_table(cases=cases, achieved=achieved, achieved_case=args.achieved_case, peak_gflops=peak_gflops, mem_bw_gbs=mem_bw_gbs)

    points = compute_points(cases=cases, achieved=achieved, achieved_case=args.achieved_case)

    plot_roofline(
        points=points,
        peak_gflops=peak_gflops,
        mem_bw_gbs=mem_bw_gbs,
        title="Laplace operator, NVIDIA Grace CPU (72 cores)",
        out_png=Path(args.out_png) if args.out_png else None,
        out_pdf=Path(args.out_pdf) if args.out_pdf else None,
    )


if __name__ == "__main__":
    main()