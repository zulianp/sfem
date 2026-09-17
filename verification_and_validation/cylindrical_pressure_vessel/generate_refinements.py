#!/usr/bin/env python3

import argparse
from pathlib import Path

import yaml

from generate_mesh import generate_mesh


def main():
    parser = argparse.ArgumentParser(description="Generate all pressure-vessel refinement meshes")
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", required=True, type=Path)
    args = parser.parse_args()

    case = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    geometry = case["geometry"]
    levels = case["refinements"]
    if sorted(level["cells"] for level in levels) != [10, 20, 40]:
        raise ValueError("pressure-vessel refinements must contain 10, 20, and 40 cells")
    for level in levels:
        cells = int(level["cells"])
        generate_mesh(
            args.output / level["id"],
            cells,
            cells,
            float(geometry["inner_radius_m"]),
            float(geometry["outer_radius_m"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
