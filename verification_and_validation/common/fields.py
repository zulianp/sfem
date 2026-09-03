"""Generation and serialization of file-backed nodal case data."""

from pathlib import Path

import numpy as np

from .raw import read_raw, typed_raw_name, write_raw


def evaluate_nodal_field(points, field):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points must be a finite two-dimensional array")
    values = field(points) if callable(field) else field
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(len(points), values, dtype=np.float64)
    if values.shape[0] != len(points):
        raise ValueError("nodal field must provide one value per point")
    if not np.all(np.isfinite(values)):
        raise ValueError("nodal field values must be finite")
    return values


def write_nodal_field(path, values, node_count=None, dtype=np.float64):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("a scalar nodal field must be one-dimensional")
    if node_count is not None and len(values) != node_count:
        raise ValueError(f"nodal field has {len(values)} entries; expected {node_count}")
    return write_raw(path, values, dtype=dtype, require_finite=True)


def read_nodal_field(path, node_count=None, dtype=np.float64):
    values = read_raw(path, dtype=dtype, require_finite=True)
    if node_count is not None and len(values) != node_count:
        raise ValueError(f"nodal field has {len(values)} entries; expected {node_count}")
    return values


def write_component_field(folder, prefix, values, dtype=np.float64):
    values = np.asarray(values)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("component field must be a finite (nodes, components) array")
    folder = Path(folder)
    paths = []
    for component in range(values.shape[1]):
        path = folder / typed_raw_name(f"{prefix}.{component}", dtype)
        write_raw(path, values[:, component], dtype=dtype, require_finite=True)
        paths.append(path)
    return paths


def read_component_field(paths, node_count=None, dtype=np.float64):
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("at least one component path is required")
    components = [read_nodal_field(path, node_count=node_count, dtype=dtype) for path in paths]
    lengths = {len(component) for component in components}
    if len(lengths) != 1:
        raise ValueError("component field streams have inconsistent lengths")
    return np.column_stack(components)


def write_boundary_values(path, mesh, nodes, field, dtype=np.float64):
    nodes = np.asarray(nodes, dtype=np.int64)
    if nodes.ndim != 1 or np.any(nodes < 0) or np.any(nodes >= mesh.n_points):
        raise ValueError("boundary node indices are invalid")
    values = evaluate_nodal_field(mesh.points[nodes], field)
    if values.ndim != 1:
        raise ValueError("boundary value file must contain one scalar component")
    write_nodal_field(path, values, node_count=len(nodes), dtype=dtype)
    return values


def write_initial_field(folder, prefix, mesh, field, dtype=np.float64):
    values = evaluate_nodal_field(mesh.points, field)
    if values.ndim == 1:
        values = values[:, None]
    return write_component_field(folder, prefix, values, dtype=dtype)
