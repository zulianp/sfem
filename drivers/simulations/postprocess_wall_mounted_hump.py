#!/usr/bin/env python3
"""Summarize wall-mounted-hump restart fields written by the M10 driver."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_required(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return np.fromfile(path, dtype=np.float64)


def summarize(output_dir: Path) -> dict[str, float | int | bool]:
    restart = output_dir / "restart"
    u0 = _read_required(restart / "u0.float64")
    u1 = _read_required(restart / "u1.float64")
    u2 = _read_required(restart / "u2.float64")
    p = _read_required(restart / "p.float64")
    marker = _read_required(restart / "boundary_marker.float64").astype(np.int64)
    if not (len(u0) == len(u1) == len(u2) == len(p) == len(marker)):
        raise ValueError("restart fields have inconsistent lengths")
    velocity_norm = np.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
    return {
        "n_nodes": int(len(u0)),
        "inlet_nodes": int(np.count_nonzero(marker == 1)),
        "outlet_nodes": int(np.count_nonzero(marker == 2)),
        "wall_nodes": int(np.count_nonzero(marker == 3)),
        "span_nodes": int(np.count_nonzero(marker == 4)),
        "u_min": float(np.min(velocity_norm)) if len(velocity_norm) else 0.0,
        "u_max": float(np.max(velocity_norm)) if len(velocity_norm) else 0.0,
        "p_min": float(np.min(p)) if len(p) else 0.0,
        "p_max": float(np.max(p)) if len(p) else 0.0,
        "has_solve_stages": (output_dir / "solve_stages.csv").exists(),
    }


def write_csv(path: Path, summary: dict[str, float | int | bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)

    summary = summarize(args.output_dir)
    if args.csv:
        write_csv(args.csv, summary)
    else:
        for key, value in summary.items():
            print("%s,%s" % (key, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
