#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.affine import generate_affine_mesh
from oracle import deformation_gradients


def main():
    parser = argparse.ArgumentParser(description="Generate the three-dimensional hyperelastic affine mesh")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--element",
        choices=("TET4", "TET10", "HEX8", "HEX27", "PROTEUS_HEX8", "PROTEUS_HEX27"),
        required=True,
    )
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--transform", choices=("aligned", "skewed"), required=True)
    args = parser.parse_args()
    metadata = generate_affine_mesh(
        args.output,
        args.element,
        (args.nx, args.ny, args.nz),
        deformation_gradients(),
        transform=args.transform,
    )
    print(
        f"Generated {metadata['element_type']} hyperelastic patch with "
        f"{metadata['interior_nodes']} interior nodes ({metadata['transform']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
