#!/usr/bin/env python3

"""The Prony bar: [0, L] x [0, 1] x [0, 1], the cross-section fixed at the unit square by
generate_bar_mesh. The free mode depends on L only (the lateral displacements are held)."""

import argparse
from pathlib import Path
import sys

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.transient import generate_bar_mesh


def main():
    parser = argparse.ArgumentParser(description="Generate the Prony modal bar")
    parser.add_argument("output", type=Path)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--length", type=float, required=True, help="the bar's length L along x")
    args = parser.parse_args()
    mesh = generate_bar_mesh(args.output, "HEX8", args.nx, args.ny, args.nz, length=args.length)
    print(f"Generated HEX8 modal bar with {mesh.n_elements} elements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
