#!/usr/bin/env python3
"""Plot extracted Stokes verification fields against node coordinates."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def read_columns(path: Path) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty field CSV: %s" % path)
    return {
        name: np.array([float(row[name]) for row in rows], dtype=np.float64)
        for name in rows[0]
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--field", default="velocity_error")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--point-size", type=float, default=8.0)
    args = parser.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/sfem_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sfem_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    (Path(os.environ["XDG_CACHE_HOME"]) / "fontconfig").mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    columns = read_columns(args.csv)
    if args.field not in columns:
        raise ValueError("field '%s' is not present in %s" % (args.field, args.csv))

    if "z" in columns:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        image = ax.scatter(
            columns["x"],
            columns["y"],
            columns["z"],
            c=columns[args.field],
            s=args.point_size,
            linewidths=0.0,
        )
        ax.set_zlabel("z")
    else:
        fig, ax = plt.subplots()
        image = ax.scatter(
            columns["x"],
            columns["y"],
            c=columns[args.field],
            s=args.point_size,
            linewidths=0.0,
        )
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(args.title or args.field)
    fig.colorbar(image, ax=ax, label=args.field)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
