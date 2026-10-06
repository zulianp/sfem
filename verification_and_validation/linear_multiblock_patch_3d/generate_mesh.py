#!/usr/bin/env python3

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.fields import write_boundary_values
from common.geometry import box_mesh
from common.mesh import write_mesh
from common.raw import write_raw
from common.sets import boundary_sides, nodeset_from_sideset, select_boundary_axis, write_nodeset
from oracle import displacement


MATERIAL = {"left_mu": 2.0, "left_lambda": 3.0, "right_mu": 5.0, "right_lambda": 7.0}


def main():
    parser = argparse.ArgumentParser(description="Generate a two-block bonded HEX8 patch")
    parser.add_argument("output", type=Path)
    parser.add_argument("--element", choices=("HEX8",), required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    args = parser.parse_args()
    if args.nx % 2:
        raise ValueError("nx must be even so the material interface lies on x=0.5")
    if args.output.exists():
        shutil.rmtree(args.output)

    mesh = box_mesh(1.0, 1.0, 1.0, args.nx, args.ny, args.nz, args.element)
    centroids = np.mean(mesh.points[mesh.elements], axis=1)
    blocks = (("left", mesh.elements[centroids[:, 0] < 0.5]), ("right", mesh.elements[centroids[:, 0] >= 0.5]))
    point_entries = []
    for component, name in enumerate(("x", "y", "z")):
        filename = f"{name}.float32"
        write_raw(args.output / filename, mesh.points[:, component], np.float32, require_finite=True)
        point_entries.append({name: filename})
    block_entries = []
    for name, elements in blocks:
        element_entries = []
        for local_node in range(8):
            filename = f"blocks/{name}/i{local_node}.int32"
            write_raw(args.output / filename, elements[:, local_node], np.int32)
            element_entries.append({f"i{local_node}": filename})
        block_entries.append({
            "name": name,
            "element_type": "HEX8",
            "elem_num_nodes": 8,
            "n_elements": int(len(elements)),
            "elements": element_entries,
        })
    metadata = {
        "spatial_dimension": 3,
        "n_blocks": 2,
        "blocks": block_entries,
        "n_nodes": mesh.n_points,
        "points": point_entries,
        "rpath": True,
    }
    (args.output / "meta.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    write_mesh(args.output / "oracle_mesh", mesh)

    boundary = nodeset_from_sideset(mesh, boundary_sides(mesh))
    interior = np.setdiff1d(np.arange(mesh.n_points, dtype=np.int64), boundary, assume_unique=True)
    right = nodeset_from_sideset(mesh, select_boundary_axis(mesh, 0, 1.0))
    write_nodeset(args.output / "sets/boundary.int32.raw", boundary)
    write_nodeset(args.output / "sets/interior.int32.raw", interior)
    write_nodeset(args.output / "sets/right.int32.raw", right)
    exact = displacement(mesh.points, MATERIAL)
    for component in range(3):
        write_boundary_values(
            args.output / f"boundary_values/displacement.{component}.float64.raw",
            mesh,
            boundary,
            exact[boundary, component],
        )
    print(f"Generated two-block HEX8 patch with {mesh.n_elements} elements and {mesh.n_points} nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
