"""Validated SFEM mesh arrays and metadata serialization."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .raw import dtype_from_path, read_raw, typed_raw_name, write_raw


ELEMENT_NODES = {
    "TRI3": 3,
    "QUAD4": 4,
    "TET4": 4,
    "HEX8": 8,
}

ELEMENT_DIMENSION = {
    "TRI3": 2,
    "QUAD4": 2,
    "TET4": 3,
    "HEX8": 3,
}

COORDINATE_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class Mesh:
    points: np.ndarray
    elements: np.ndarray
    element_type: str

    def __post_init__(self):
        element_type = str(self.element_type).upper()
        if element_type not in ELEMENT_NODES:
            raise ValueError(f"unsupported element type: {self.element_type}")

        points = np.ascontiguousarray(self.points, dtype=np.float64)
        elements = np.ascontiguousarray(self.elements, dtype=np.int64)
        dimension = ELEMENT_DIMENSION[element_type]
        nodes_per_element = ELEMENT_NODES[element_type]
        if points.ndim != 2 or points.shape[1] != dimension or not len(points):
            raise ValueError(f"{element_type} points must have shape (n, {dimension})")
        if not np.all(np.isfinite(points)):
            raise ValueError("mesh points must be finite")
        if elements.ndim != 2 or elements.shape[1] != nodes_per_element or not len(elements):
            raise ValueError(f"{element_type} elements must have shape (n, {nodes_per_element})")
        if np.any(elements < 0) or np.any(elements >= len(points)):
            raise ValueError("mesh connectivity contains an out-of-range node")
        if np.any(np.diff(np.sort(elements, axis=1), axis=1) == 0):
            raise ValueError("mesh connectivity contains a repeated node")

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "element_type", element_type)

    @property
    def dimension(self):
        return self.points.shape[1]

    @property
    def n_points(self):
        return self.points.shape[0]

    @property
    def n_elements(self):
        return self.elements.shape[0]


def _metadata_streams(metadata, key, expected_names):
    streams = metadata.get(key)
    if streams is None:
        return None
    if not isinstance(streams, list) or len(streams) != len(expected_names):
        raise ValueError(f"mesh metadata '{key}' must contain {len(expected_names)} streams")
    names = []
    for index, entry in enumerate(streams):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"mesh metadata '{key}[{index}]' must be a one-entry mapping")
        logical_name, filename = next(iter(entry.items()))
        if logical_name != expected_names[index] or not isinstance(filename, str) or not filename:
            raise ValueError(f"invalid mesh metadata stream '{key}[{index}]'")
        names.append(filename)
    return names


def _find_stream(folder, stem, default_dtype):
    candidates = sorted(path for path in folder.glob(f"{stem}.*") if path.is_file())
    direct = folder / f"{stem}.raw"
    if direct.is_file() and direct not in candidates:
        candidates.append(direct)
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one stream for '{stem}' in {folder}, found {len(candidates)}")
    return candidates[0].name, dtype_from_path(candidates[0], default=default_dtype)


def read_mesh(folder, geometry_dtype=np.float32, index_dtype=np.int32):
    folder = Path(folder)
    metadata_path = folder / "meta.yaml"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"mesh metadata must be a mapping: {metadata_path}")

    element_type = str(metadata.get("element_type", "")).upper()
    if element_type not in ELEMENT_NODES:
        raise ValueError(f"unsupported or missing element_type in {metadata_path}")
    dimension = ELEMENT_DIMENSION[element_type]
    nodes_per_element = ELEMENT_NODES[element_type]

    point_names = _metadata_streams(metadata, "points", COORDINATE_NAMES[:dimension])
    if point_names is None:
        point_specs = [_find_stream(folder, COORDINATE_NAMES[d], geometry_dtype) for d in range(dimension)]
    else:
        point_specs = [(name, dtype_from_path(folder / name, geometry_dtype)) for name in point_names]

    element_names = _metadata_streams(
        metadata, "elements", tuple(f"i{index}" for index in range(nodes_per_element))
    )
    if element_names is None:
        element_specs = [_find_stream(folder, f"i{d}", index_dtype) for d in range(nodes_per_element)]
    else:
        element_specs = [(name, dtype_from_path(folder / name, index_dtype)) for name in element_names]

    point_streams = [read_raw(folder / name, dtype=dtype, require_finite=True) for name, dtype in point_specs]
    element_streams = [read_raw(folder / name, dtype=dtype) for name, dtype in element_specs]
    point_lengths = {len(stream) for stream in point_streams}
    element_lengths = {len(stream) for stream in element_streams}
    if len(point_lengths) != 1 or len(element_lengths) != 1:
        raise ValueError(f"mesh streams have inconsistent lengths in {folder}")

    mesh = Mesh(np.column_stack(point_streams), np.column_stack(element_streams), element_type)
    if "n_points" in metadata and metadata["n_points"] != mesh.n_points:
        raise ValueError(f"mesh metadata n_points does not match streams in {folder}")
    if "n_elements" in metadata and metadata["n_elements"] != mesh.n_elements:
        raise ValueError(f"mesh metadata n_elements does not match streams in {folder}")
    return mesh


def write_mesh(folder, mesh, geometry_dtype=np.float32, index_dtype=np.int32):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    point_entries = []
    for component in range(mesh.dimension):
        name = typed_raw_name(COORDINATE_NAMES[component], geometry_dtype)
        write_raw(folder / name, mesh.points[:, component], geometry_dtype, require_finite=True)
        point_entries.append({COORDINATE_NAMES[component]: name})

    element_entries = []
    for local_node in range(mesh.elements.shape[1]):
        name = typed_raw_name(f"i{local_node}", index_dtype)
        write_raw(folder / name, mesh.elements[:, local_node], index_dtype)
        element_entries.append({f"i{local_node}": name})

    metadata = {
        "spatial_dimension": mesh.dimension,
        "element_type": mesh.element_type,
        "elem_num_nodes": mesh.elements.shape[1],
        "n_elements": mesh.n_elements,
        "n_points": mesh.n_points,
        "elements": element_entries,
        "points": point_entries,
        "rpath": True,
    }
    metadata_path = folder / "meta.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return metadata_path
