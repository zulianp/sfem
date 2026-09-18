"""Shared mesh generation and result handling for affine exact cases."""

from pathlib import Path
import shutil

import numpy as np
import yaml

from .fields import write_boundary_values
from .geometry import (
    box_mesh,
    promote_simplex_mesh,
    proteus_rectangle_mesh,
    rectangle_mesh,
    tensor_product_mesh,
)
from .mesh import Mesh, read_mesh, write_mesh
from .metrics import relative_l2_error
from .raw import dtype_from_path, read_raw, write_raw
from .reporting import make_check
from .sets import (
    Sideset,
    boundary_sides,
    nodeset_from_sideset,
    read_nodeset,
    read_sideset,
    side_nodes,
    validate_sideset_orientation,
    write_nodeset,
    write_sideset,
)


TWO_DIMENSIONAL_ELEMENTS = {"TRI3", "TRI6", "QUAD4", "PROTEUS_QUAD4"}
MODE_COMPLETION_MARKER = "SFEM_AFFINE_MODE_COMPLETE"


def _transform(dimension, kind):
    if kind == "aligned":
        return np.eye(dimension)
    if kind != "skewed":
        raise ValueError("transform must be 'aligned' or 'skewed'")
    if dimension == 2:
        return np.asarray(((1.0, 0.25), (0.125, 1.0)))
    if dimension == 3:
        return np.asarray(((1.0, 0.125, -0.0625), (0.0625, 1.0, 0.125), (0.03125, -0.0625, 1.0)))
    raise ValueError("affine cases support only two or three dimensions")


def generate_affine_mesh(output, element_type, resolution, deformation_gradients, transform="aligned"):
    """Generate an affine unit domain, its sets, and all prescribed boundary fields."""

    output = Path(output)
    if output.exists():
        shutil.rmtree(output)

    element_type = str(element_type).upper()
    dimension = 2 if element_type in TWO_DIMENSIONAL_ELEMENTS else 3
    if dimension == 2:
        nx, ny = (int(value) for value in resolution)
        if element_type == "TRI6":
            base_mesh = promote_simplex_mesh(rectangle_mesh(1.0, 1.0, nx, ny, "TRI3"), "TRI6")
        elif element_type == "PROTEUS_QUAD4":
            base_mesh = proteus_rectangle_mesh(1.0, 1.0, nx, ny)
        else:
            base_mesh = rectangle_mesh(1.0, 1.0, nx, ny, element_type=element_type)
    else:
        nx, ny, nz = (int(value) for value in resolution)
        if element_type == "TET10":
            base_mesh = promote_simplex_mesh(box_mesh(1.0, 1.0, 1.0, nx, ny, nz, "TET4"), "TET10")
        elif element_type in ("HEX27", "PROTEUS_HEX8", "PROTEUS_HEX27"):
            base_mesh = tensor_product_mesh(1.0, 1.0, 1.0, nx, ny, nz, element_type)
        else:
            base_mesh = box_mesh(1.0, 1.0, 1.0, nx, ny, nz, element_type=element_type)

    coordinate_transform = _transform(dimension, transform)
    if np.linalg.det(coordinate_transform) <= 0:
        raise ValueError("coordinate transform must preserve element orientation")
    points = base_mesh.points @ coordinate_transform.T
    mesh = Mesh(points, base_mesh.elements, element_type)
    write_mesh(output, mesh)
    mesh = read_mesh(output)

    exterior = boundary_sides(mesh)
    orientation = validate_sideset_orientation(mesh, exterior)
    exterior_nodes = nodeset_from_sideset(mesh, exterior)
    all_nodes = np.arange(mesh.n_points, dtype=np.int64)
    interior_nodes = np.setdiff1d(all_nodes, exterior_nodes, assume_unique=True)
    if not len(interior_nodes):
        raise ValueError("affine verification meshes require at least one interior node")

    exterior_side_nodes = side_nodes(mesh, exterior)
    on_reaction_face = np.all(np.isclose(base_mesh.points[exterior_side_nodes, 0], 1.0), axis=1)
    reaction_sides = Sideset(exterior.parent[on_reaction_face], exterior.local_side[on_reaction_face])
    reaction_orientation = validate_sideset_orientation(mesh, reaction_sides)
    reaction_nodes = nodeset_from_sideset(mesh, reaction_sides)

    sets = output / "sets"
    write_nodeset(sets / "boundary.int32.raw", exterior_nodes)
    write_nodeset(sets / "interior.int32.raw", interior_nodes)
    write_nodeset(sets / "reaction_face.int32.raw", reaction_nodes)
    write_sideset(sets / "reaction_face_sides", mesh, reaction_sides)

    values_dir = output / "boundary_values"
    for mode, deformation in deformation_gradients.items():
        deformation = np.asarray(deformation, dtype=np.float64)
        if deformation.shape != (dimension, dimension):
            raise ValueError(f"mode {mode!r} has deformation gradient shape {deformation.shape}")
        gradient = deformation - np.eye(dimension)
        values = mesh.points @ gradient.T
        for component in range(dimension):
            write_raw(
                output / "initial_values" / f"{mode}.{component}.float64.raw",
                values[:, component],
                np.float64,
                require_finite=True,
            )
            write_boundary_values(
                values_dir / f"{mode}.{component}.float64.raw",
                mesh,
                exterior_nodes,
                values[exterior_nodes, component],
            )

    metadata = {
        "dimension": dimension,
        "element_type": element_type,
        "resolution": [int(value) for value in resolution],
        "transform": transform,
        "coordinate_transform": coordinate_transform.tolist(),
        "modes": list(deformation_gradients),
        "boundary_nodes": int(len(exterior_nodes)),
        "interior_nodes": int(len(interior_nodes)),
        "reaction_face_nodes": int(len(reaction_nodes)),
        "minimum_boundary_orientation_cosine": orientation["minimum_orientation_cosine"],
        "minimum_reaction_face_orientation_cosine": reaction_orientation["minimum_orientation_cosine"],
    }
    (output / "generation.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return metadata


def load_affine_mesh(output):
    output = Path(output)
    return {
        "mesh": read_mesh(output),
        "boundary_nodes": read_nodeset(output / "sets" / "boundary.int32.raw"),
        "interior_nodes": read_nodeset(output / "sets" / "interior.int32.raw"),
        "reaction_nodes": read_nodeset(output / "sets" / "reaction_face.int32.raw"),
        "reaction_sides": read_sideset(output / "sets" / "reaction_face_sides"),
    }


def _find_component(folder, name, component):
    candidates = []
    for path in sorted(Path(folder).glob(f"{name}.{component}.*")):
        if not path.is_file():
            continue
        try:
            dtype_from_path(path)
        except ValueError:
            continue
        candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"expected one {name} component {component} in {folder}, found {len(candidates)}")
    return candidates[0]


def read_component_output(folder, name, dimension, node_count):
    components = []
    for component in range(dimension):
        path = _find_component(folder, name, component)
        values = read_raw(path, require_finite=True)
        if len(values) != node_count:
            raise ValueError(f"{path} contains {len(values)} entries; expected {node_count}")
        components.append(values.astype(np.float64, copy=False))
    return np.column_stack(components)


def read_mode_solution(solution_root, mode, driver_kind, dimension, node_count):
    solution = Path(solution_root) / mode
    if driver_kind == "linear":
        field_dir = solution
        displacement_name = "x"
    elif driver_kind == "hyperelastic":
        field_dir = solution / "out"
        displacement_name = "disp"
    else:
        raise ValueError(f"unsupported affine driver kind: {driver_kind}")

    displacement = read_component_output(field_dir, displacement_name, dimension, node_count)
    reaction = read_component_output(field_dir, "material_reaction", dimension, node_count)
    quantities_path = solution / "quantities.yaml"
    quantities = yaml.safe_load(quantities_path.read_text(encoding="utf-8"))
    if driver_kind == "linear":
        objective = quantities.get("material_objective")
    else:
        history = quantities.get("material_objective_history")
        objective = history[-1].get("value") if isinstance(history, list) and history else None
    if not isinstance(objective, (int, float)) or not np.isfinite(float(objective)):
        raise ValueError(f"missing finite material objective in {quantities_path}")
    return displacement, reaction, float(objective)


def affine_checks(
    mode,
    mesh,
    interior_nodes,
    reaction_nodes,
    observed_displacement,
    observed_reaction,
    observed_energy,
    expected_deformation,
    expected_stress,
    expected_energy,
    tolerances,
    oracle,
    deformation_jacobians=None,
):
    expected_displacement = mesh.points @ (np.asarray(expected_deformation) - np.eye(mesh.dimension)).T
    expected_resultant = np.asarray(expected_stress) @ _reaction_area_vector(mesh, reaction_nodes)
    observed_resultant = np.sum(observed_reaction[reaction_nodes], axis=0)

    displacement_error = relative_l2_error(observed_displacement, expected_displacement)
    residual_scale = max(float(np.linalg.norm(expected_resultant)), 1.0e-14)
    free_residual = float(np.linalg.norm(observed_reaction[interior_nodes])) / residual_scale
    energy_error = abs(observed_energy - expected_energy) / max(abs(expected_energy), 1.0e-14)
    reaction_error = relative_l2_error(observed_resultant, expected_resultant)

    checks = [
        make_check(
            f"{mode}_displacement_relative_l2",
            displacement_error,
            0.0,
            displacement_error,
            tolerances[f"{mode}_displacement_relative_l2"],
            "1",
            oracle,
        ),
        make_check(
            f"{mode}_free_residual_normalized",
            free_residual,
            0.0,
            free_residual,
            tolerances[f"{mode}_free_residual_normalized"],
            "1",
            oracle,
        ),
        make_check(
            f"{mode}_energy_relative",
            observed_energy,
            expected_energy,
            energy_error,
            tolerances[f"{mode}_energy_relative"],
            "1",
            oracle,
        ),
        make_check(
            f"{mode}_reaction_relative_l2",
            reaction_error,
            0.0,
            reaction_error,
            tolerances[f"{mode}_reaction_relative_l2"],
            "1",
            oracle,
        ),
    ]
    diagnostics = {
        "observed_reaction_resultant": observed_resultant.tolist(),
        "expected_reaction_resultant": expected_resultant.tolist(),
        "observed_energy": observed_energy,
        "expected_energy": expected_energy,
        "free_residual_norm": float(np.linalg.norm(observed_reaction[interior_nodes])),
    }

    if deformation_jacobians is not None:
        minimum_jacobian = float(np.min(deformation_jacobians))
        deficit = max(0.0, np.finfo(np.float64).eps - minimum_jacobian)
        checks.append(
            make_check(
                f"{mode}_minimum_jacobian_deficit",
                minimum_jacobian,
                0.0,
                deficit,
                tolerances[f"{mode}_minimum_jacobian_deficit"],
                "1",
                oracle,
            )
        )
        diagnostics["minimum_deformation_jacobian"] = minimum_jacobian

    return checks, diagnostics


def _reaction_area_vector(mesh, reaction_nodes):
    """Recover the oriented reference area vector of the generated x-plus face."""

    artifacts = load_affine_mesh_from_mesh(mesh, reaction_nodes)
    return artifacts


def load_affine_mesh_from_mesh(mesh, reaction_nodes):
    exterior = boundary_sides(mesh)
    candidates = side_nodes(mesh, exterior)
    reaction_nodes = np.asarray(reaction_nodes, dtype=np.int64)
    selected = np.all(np.isin(candidates, reaction_nodes), axis=1)
    sides = Sideset(exterior.parent[selected], exterior.local_side[selected])
    if sides.size == 0:
        raise ValueError("reaction nodes do not identify a boundary face")
    from .sets import surface_geometry

    return np.sum(surface_geometry(mesh, sides).area_vectors, axis=0)
