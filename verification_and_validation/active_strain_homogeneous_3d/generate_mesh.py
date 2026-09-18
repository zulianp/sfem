#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.affine import generate_affine_mesh
from common.mesh import read_mesh
from common.raw import write_raw
from oracle import active_gradient, deformation_gradients


def main():
    parser = argparse.ArgumentParser(description="Generate the active-strain homogeneous cube")
    parser.add_argument("output", type=Path)
    parser.add_argument("--element", choices=("HEX8",), required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    args = parser.parse_args()

    metadata = generate_affine_mesh(
        args.output,
        args.element,
        (args.nx, args.ny, args.nz),
        deformation_gradients(),
        transform="aligned",
    )
    mesh = read_mesh(args.output)
    field = np.tile(active_gradient().reshape(-1), mesh.n_elements)
    write_raw(args.output / "fields" / "Fa.float64.raw", field, np.float64, require_finite=True)
    print(f"Generated {metadata['element_type']} active-strain cube with {mesh.n_elements} elements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
