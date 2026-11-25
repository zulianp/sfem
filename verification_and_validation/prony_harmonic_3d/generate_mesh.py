#!/usr/bin/env python3

"""The harmonically driven bar: [0, L] x [0, 1] x [0, 1], the cross-section fixed at the unit
square by generate_bar_mesh, so the right face's reaction resultant is P_11. The strain is
homogeneous, so L does not enter the response."""

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
    parser = argparse.ArgumentParser(description="Generate the harmonically driven Prony bar")
    parser.add_argument("output", type=Path)
    parser.add_argument("--element", required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--length", type=float, required=True, help="the bar's length L along x")
    parser.add_argument("--amplitude", type=float, required=True, help="the strain amplitude eps0")
    args = parser.parse_args()
    mesh = generate_bar_mesh(args.output, args.element, args.nx, args.ny, args.nz, length=args.length)
    # The right face's axial displacement at unit profile: eps0 x (= eps0 L), the homogeneous field's.
    right = read_raw(args.output / "sets" / "right.int32.raw", dtype=np.int32)
    values = args.amplitude * mesh.points[right, 0]
    write_raw(args.output / "boundary_values" / "right_x.float64.raw", values, np.float64, require_finite=True)
    print(f"Generated {mesh.element_type} harmonic Prony bar with {mesh.n_elements} elements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
