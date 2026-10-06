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
from common.mechanics import element_kinematics, integrate_boundary_traction
from common.mesh import write_mesh
from common.raw import write_raw
from common.sets import (
    Sideset, boundary_sides, side_nodes, surface_geometry,
    validate_sideset_orientation, write_nodeset, write_sideset,
)
from oracle import octant_pressure_resultant


def radius_sides(mesh, radius):
    boundary = boundary_sides(mesh)
    nodes = side_nodes(mesh, boundary)
    selected = np.all(
        np.isclose(np.linalg.norm(mesh.points[nodes], axis=-1), radius, rtol=0.0, atol=1.0e-10),
        axis=1,
    )
    return Sideset(boundary.parent[selected], boundary.local_side[selected])


def main():
    parser = argparse.ArgumentParser(description="Generate pressure-loaded Lame spherical octants")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    element = config["selected_variant"]["element"]
    inner_radius = float(config["geometry"]["inner_radius_m"])
    outer_radius = float(config["geometry"]["outer_radius_m"])
    pressure = float(config["loading"]["inner_pressure_mpa"])
    expected_resultant = octant_pressure_resultant(inner_radius, pressure)
    diagnostics = {}

    for level in config["refinements"]:
        level_id = level["id"]
        level_dir = args.output / level_id
        mesh = spherical_shell_octant_mesh(
            inner_radius, outer_radius, int(level["radial_cells"]),
            int(level["surface_frequency"]), element,
        )
        element_kinematics(mesh, np.zeros_like(mesh.points))
        write_mesh(level_dir, mesh)
        inner = radius_sides(mesh, inner_radius)
        outer = radius_sides(mesh, outer_radius)
        if not inner.size or not outer.size:
            raise ValueError(f"{level_id}: spherical boundary was not found")
        inner_orientation = validate_sideset_orientation(mesh, inner)
        outer_orientation = validate_sideset_orientation(mesh, outer)
        write_sideset(level_dir / "sets" / "inner", mesh, inner)
        write_sideset(level_dir / "sets" / "outer", mesh, outer)
        for axis, name in enumerate(("x0", "y0", "z0")):
            nodes = np.flatnonzero(np.abs(mesh.points[:, axis]) <= 1.0e-12)
            if not len(nodes):
                raise ValueError(f"{level_id}: symmetry plane {name} is empty")
            write_nodeset(level_dir / "sets" / f"{name}.int32.raw", nodes)

        geometry = surface_geometry(mesh, inner)
        traction = -pressure * geometry.normals
        for axis in range(3):
            write_raw(
                level_dir / "traction" / f"inner.{axis}.float64.raw",
                traction[:, axis], np.float64, require_finite=True,
            )
        resultant = integrate_boundary_traction(mesh, inner, traction)
        area = float(np.sum(geometry.measures))
        expected_area = 0.5 * np.pi * inner_radius ** 2
        area_error = abs(area / expected_area - 1.0)
        if area_error > 0.06:
            raise ValueError(f"{level_id}: inner surface area error {area_error:.6g} exceeds 6%")
        diagnostics[level_id] = {
            "element_count": mesh.n_elements,
            "point_count": mesh.n_points,
            "inner_face_count": inner.size,
            "inner_area_m2": area,
            "inner_area_relative_error": area_error,
            "applied_pressure_resultant_mpa_m2": resultant.tolist(),
            "applied_resultant_relative_l2": float(
                np.linalg.norm(resultant - expected_resultant) / np.linalg.norm(expected_resultant)
            ),
            "inner_orientation": inner_orientation,
            "outer_orientation": outer_orientation,
        }
        print(f"Generated {level_id}: {mesh.n_elements} {element} elements; inner area={area:.6g}")
    (args.output / "geometry_diagnostics.yaml").write_text(
        yaml.safe_dump(diagnostics, sort_keys=False), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
