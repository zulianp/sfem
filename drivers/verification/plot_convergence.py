#!/usr/bin/env python3
"""Plot Stokes convergence tables produced by run_stokes_convergence.py."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/sfem_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sfem_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    (Path(os.environ["XDG_CACHE_HOME"]) / "fontconfig").mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    rows = sorted(read_rows(args.csv), key=lambda row: float(row["h"]), reverse=True)
    h = np.array([float(row["h"]) for row in rows])
    velocity = np.array([float(row["velocity_l2_abs"]) for row in rows])
    pressure = np.array([float(row["pressure_l2_abs"]) for row in rows])

    fig, ax = plt.subplots()
    ax.loglog(h, velocity, "o-", label="velocity L2")
    ax.loglog(h, pressure, "s-", label="pressure L2")
    ax.invert_xaxis()
    ax.set_xlabel("h")
    ax.set_ylabel("error")
    ax.grid(True, which="both")
    ax.legend()
    if args.title:
        ax.set_title(args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
