#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.geometry import annular_sector_mesh
from common.mechanics import element_kinematics
from common.mesh import write_mesh
from common.raw import write_raw
from common.sets import (
    Sideset, boundary_sides, side_nodes, surface_geometry,
    validate_sideset_orientation, write_nodeset, write_sideset,
)
from oracle import displacement


def radius_sides(mesh, radius, atol=1.0e-10):
    boundary = boundary_sides(mesh)
    nodes = side_nodes(mesh, boundary)
    selected = np.all(
        np.isclose(np.linalg.norm(mesh.points[nodes], axis=-1), radius, rtol=0.0, atol=atol),
        axis=1,
    )
    return Sideset(boundary.parent[selected], boundary.local_side[selected])


def main():
    parser = argparse.ArgumentParser(description="Generate Kirsch quarter-annulus refinements")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    element = config["selected_variant"]["element"]
    a = float(config["geometry"]["hole_radius_m"])
    outer = float(config["geometry"]["outer_radius_m"])
    tension = float(config["loading"]["remote_tension_mpa"])
    mu = float(config["material"]["mu"])
    lmbda = float(config["material"]["lambda"])
    diagnostics = {}

    for level in config["refinements"]:
        level_id = level["id"]
        level_dir = args.output / level_id
        mesh = annular_sector_mesh(
            a, outer, int(level["radial_cells"]), int(level["angular_cells"]),
            element_type=element,
        )
        element_kinematics(mesh, np.zeros_like(mesh.points))
        write_mesh(level_dir, mesh)
        inner = radius_sides(mesh, a)
        exterior = radius_sides(mesh, outer)
        if inner.size != level["angular_cells"] or exterior.size != level["angular_cells"]:
            raise ValueError(f"{level_id}: incomplete inner or outer arc")
        inner_orientation = validate_sideset_orientation(mesh, inner)
        outer_orientation = validate_sideset_orientation(mesh, exterior)
        write_sideset(level_dir / "sets" / "inner", mesh, inner)
        write_sideset(level_dir / "sets" / "outer", mesh, exterior)

        radii = np.linalg.norm(mesh.points, axis=1)
        outer_nodes = np.flatnonzero(np.isclose(radii, outer, rtol=0.0, atol=1.0e-10))
        axis_x = np.flatnonzero((np.abs(mesh.points[:, 1]) <= 1.0e-12) & (radii < outer - 1.0e-10))
        axis_y = np.flatnonzero((np.abs(mesh.points[:, 0]) <= 1.0e-12) & (radii < outer - 1.0e-10))
        if len(axis_x) != level["radial_cells"] or len(axis_y) != level["radial_cells"]:
            raise ValueError(f"{level_id}: incomplete symmetry axes")
        write_nodeset(level_dir / "sets" / "outer.int32.raw", outer_nodes)
        write_nodeset(level_dir / "sets" / "axis_x.int32.raw", axis_x)
        write_nodeset(level_dir / "sets" / "axis_y.int32.raw", axis_y)
        values = displacement(mesh.points[outer_nodes], a, tension, mu, lmbda)
        for component in range(2):
            write_raw(
                level_dir / "boundary_values" / f"outer.{component}.float64.raw",
                values[:, component], np.float64, require_finite=True,
            )
        inner_length = float(np.sum(surface_geometry(mesh, inner).measures))
        diagnostics[level_id] = {
            "element_count": mesh.n_elements,
            "point_count": mesh.n_points,
            "inner_arc_length_m": inner_length,
            "inner_arc_length_relative_error": abs(inner_length / (0.5 * np.pi * a) - 1.0),
            "inner_orientation": inner_orientation,
            "outer_orientation": outer_orientation,
        }
        print(f"Generated {level_id}: {mesh.n_elements} {element} elements")
    (args.output / "geometry_diagnostics.yaml").write_text(
        yaml.safe_dump(diagnostics, sort_keys=False), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
