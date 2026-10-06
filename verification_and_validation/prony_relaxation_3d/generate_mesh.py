#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.raw import read_raw, write_raw
from common.transient import generate_bar_mesh


def main():
    parser = argparse.ArgumentParser(description="Generate the prescribed Prony relaxation bar")
    parser.add_argument("output", type=Path)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--strain", type=float, required=True)
    args = parser.parse_args()
    mesh = generate_bar_mesh(args.output, "HEX8", args.nx, args.ny, args.nz)
    right = read_raw(args.output / "sets" / "right.int32.raw", dtype=np.int32)
    values = args.strain * mesh.points[right, 0]
    write_raw(args.output / "boundary_values" / "right_x.float64.raw", values, np.float64, require_finite=True)
    print(f"Generated HEX8 Prony bar with {mesh.n_elements} elements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
