#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.transient import generate_bar_mesh


def main():
    parser = argparse.ArgumentParser(description="Generate the 2D Kelvin-Voigt creep strip")
    parser.add_argument("output", type=Path)
    parser.add_argument("--element", choices=("TRI3", "QUAD4"), required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    args = parser.parse_args()
    mesh = generate_bar_mesh(args.output, args.element, args.nx, args.ny)
    print(f"Generated {mesh.element_type} creep strip with {mesh.n_elements} elements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
