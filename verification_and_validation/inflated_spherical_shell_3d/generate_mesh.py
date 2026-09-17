#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.geometry import spherical_shell_octant_mesh
from common.mechanics import element_kinematics
from common.mesh import write_mesh
from common.sets import (
    Sideset, boundary_sides, side_nodes, surface_geometry,
    validate_sideset_orientation, write_nodeset, write_sideset,
)


def radius_sides(mesh, radius):
    boundary = boundary_sides(mesh)
    nodes = side_nodes(mesh, boundary)
    selected = np.all(
        np.isclose(np.linalg.norm(mesh.points[nodes], axis=-1), radius, rtol=0.0, atol=1.0e-10),
        axis=1,
    )
    return Sideset(boundary.parent[selected], boundary.local_side[selected])


def main():
    parser = argparse.ArgumentParser(description="Generate finite-strain spherical octant refinements")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    a = float(config["geometry"]["inner_radius_m"])
    b = float(config["geometry"]["outer_radius_m"])
    element = config["selected_variant"]["element"]
    diagnostics = {}
    for level in config["refinements"]:
        level_id = level["id"]
        folder = args.output / level_id
        mesh = spherical_shell_octant_mesh(
            a, b, int(level["radial_cells"]), int(level["surface_frequency"]), element,
        )
        element_kinematics(mesh, np.zeros_like(mesh.points))
        write_mesh(folder, mesh)
        inner = radius_sides(mesh, a)
        outer = radius_sides(mesh, b)
        if not inner.size or not outer.size:
            raise ValueError(f"{level_id}: missing inner or outer spherical surface")
        inner_orientation = validate_sideset_orientation(mesh, inner)
        outer_orientation = validate_sideset_orientation(mesh, outer)
        write_sideset(folder / "sets" / "inner", mesh, inner)
        write_sideset(folder / "sets" / "outer", mesh, outer)
        for axis, name in enumerate(("x0", "y0", "z0")):
            nodes = np.flatnonzero(np.abs(mesh.points[:, axis]) <= 1.0e-12)
            if not len(nodes):
                raise ValueError(f"{level_id}: empty symmetry plane {name}")
            write_nodeset(folder / "sets" / f"{name}.int32.raw", nodes)
        inner_area = float(np.sum(surface_geometry(mesh, inner).measures))
        exact_area = 0.5 * np.pi * a ** 2
        area_error = abs(inner_area / exact_area - 1.0)
        if area_error > 0.06:
            raise ValueError(f"{level_id}: inner pressure area error {area_error:.6g} exceeds 6%")
        diagnostics[level_id] = {
            "element_count": mesh.n_elements,
            "point_count": mesh.n_points,
            "inner_face_count": inner.size,
            "inner_area_m2": inner_area,
            "inner_area_relative_error": area_error,
            "inner_orientation": inner_orientation,
            "outer_orientation": outer_orientation,
        }
        print(f"Generated {level_id}: {mesh.n_elements} {element} elements; inner area={inner_area:.6g}")
    (args.output / "geometry_diagnostics.yaml").write_text(
        yaml.safe_dump(diagnostics, sort_keys=False), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
