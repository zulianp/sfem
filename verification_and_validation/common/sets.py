"""SFEM-compatible sidesets, nodesets, and boundary geometry checks."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .mesh import Mesh
from .raw import read_raw, write_raw


LOCAL_SIDES = {
    "TRI3": ((0, 1), (1, 2), (2, 0)),
    "QUAD4": ((0, 1), (1, 2), (2, 3), (3, 0)),
    "TET4": ((0, 1, 3), (1, 2, 3), (0, 3, 2), (0, 2, 1)),
    "HEX8": (
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
        (3, 2, 1, 0),
        (4, 5, 6, 7),
    ),
}

SIDE_ELEMENT_TYPE = {
    "TRI3": "EDGESHELL2",
    "QUAD4": "EDGESHELL2",
    "TET4": "TRISHELL3",
    "HEX8": "QUADSHELL4",
}


@dataclass(frozen=True)
class Sideset:
    parent: np.ndarray
    local_side: np.ndarray

    def __post_init__(self):
        parent = np.ascontiguousarray(self.parent, dtype=np.int64)
        local_side = np.ascontiguousarray(self.local_side, dtype=np.int16)
        if parent.ndim != 1 or local_side.ndim != 1 or len(parent) != len(local_side):
            raise ValueError("sideset parent and local_side arrays must be one-dimensional and equally sized")
        if np.any(parent < 0) or np.any(local_side < 0):
            raise ValueError("sideset indices must be non-negative")
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "local_side", local_side)

    @property
    def size(self):
        return len(self.parent)


@dataclass(frozen=True)
class SurfaceGeometry:
    nodes: np.ndarray
    centroids: np.ndarray
    area_vectors: np.ndarray
    normals: np.ndarray
    measures: np.ndarray


def validate_sideset(mesh, sideset):
    if mesh.element_type not in LOCAL_SIDES:
        raise ValueError(f"unsupported sideset element type: {mesh.element_type}")
    if np.any(sideset.parent >= mesh.n_elements):
        raise ValueError("sideset contains an out-of-range parent element")
    if np.any(sideset.local_side >= len(LOCAL_SIDES[mesh.element_type])):
        raise ValueError("sideset contains an out-of-range local side")


def side_nodes(mesh, sideset):
    validate_sideset(mesh, sideset)
    table = np.asarray(LOCAL_SIDES[mesh.element_type], dtype=np.int64)
    local_nodes = table[sideset.local_side]
    return np.take_along_axis(mesh.elements[sideset.parent], local_nodes, axis=1)


def boundary_sides(mesh):
    table = LOCAL_SIDES.get(mesh.element_type)
    if table is None:
        raise ValueError(f"unsupported sideset element type: {mesh.element_type}")
    occurrences = {}
    for parent, element in enumerate(mesh.elements):
        for local_side, local_nodes in enumerate(table):
            nodes = tuple(int(element[index]) for index in local_nodes)
            key = tuple(sorted(nodes))
            occurrences.setdefault(key, []).append((parent, local_side))
    non_manifold = sum(len(items) > 2 for items in occurrences.values())
    if non_manifold:
        raise ValueError(f"mesh contains {non_manifold} non-manifold sides")
    boundary = sorted(items[0] for items in occurrences.values() if len(items) == 1)
    if not boundary:
        return Sideset(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int16))
    parent, local_side = np.asarray(boundary, dtype=np.int64).T
    return Sideset(parent, local_side)


def nodeset_from_sideset(mesh, sideset):
    nodes = side_nodes(mesh, sideset)
    return np.unique(nodes.ravel())


def surface_geometry(mesh, sideset):
    nodes = side_nodes(mesh, sideset)
    coordinates = mesh.points[nodes]
    centroids = np.mean(coordinates, axis=1)
    if mesh.dimension == 2:
        tangents = coordinates[:, 1] - coordinates[:, 0]
        area_vectors = np.column_stack((tangents[:, 1], -tangents[:, 0]))
        measures = np.linalg.norm(tangents, axis=1)
    elif coordinates.shape[1:] == (3, 3):
        area_vectors = 0.5 * np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0])
        measures = np.linalg.norm(area_vectors, axis=1)
    elif coordinates.shape[1:] == (4, 3):
        first = 0.5 * np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0])
        second = 0.5 * np.cross(coordinates[:, 2] - coordinates[:, 0], coordinates[:, 3] - coordinates[:, 0])
        area_vectors = first + second
        measures = np.linalg.norm(first, axis=1) + np.linalg.norm(second, axis=1)
    else:
        raise ValueError("unsupported boundary topology")
    if np.any(measures <= np.finfo(np.float64).eps):
        raise ValueError("sideset contains a degenerate side")
    normal_magnitudes = np.linalg.norm(area_vectors, axis=1)
    if np.any(normal_magnitudes <= np.finfo(np.float64).eps):
        raise ValueError("sideset contains a side with an undefined normal")
    normals = area_vectors / normal_magnitudes[:, None]
    return SurfaceGeometry(nodes, centroids, area_vectors, normals, measures)


def orientation_cosines(mesh, sideset):
    geometry = surface_geometry(mesh, sideset)
    element_centroids = np.mean(mesh.points[mesh.elements[sideset.parent]], axis=1)
    outward = geometry.centroids - element_centroids
    lengths = np.linalg.norm(outward, axis=1)
    if np.any(lengths <= np.finfo(np.float64).eps):
        raise ValueError("cannot determine outward direction for a side")
    return np.einsum("ij,ij->i", geometry.normals, outward / lengths[:, None])


def validate_sideset_orientation(mesh, sideset, minimum_cosine=0.0):
    cosines = orientation_cosines(mesh, sideset)
    invalid = np.flatnonzero(cosines <= minimum_cosine)
    if len(invalid):
        raise ValueError(
            f"sideset has {len(invalid)} inward or ambiguous sides; minimum orientation cosine={np.min(cosines):.6g}"
        )
    return {
        "side_count": sideset.size,
        "minimum_orientation_cosine": float(np.min(cosines)) if len(cosines) else None,
    }


def select_boundary_sides(mesh, predicate):
    sideset = boundary_sides(mesh)
    geometry = surface_geometry(mesh, sideset)
    selected = np.asarray(predicate(geometry.centroids, geometry.normals), dtype=bool)
    if selected.shape != (sideset.size,):
        raise ValueError("boundary predicate must return one boolean per side")
    return Sideset(sideset.parent[selected], sideset.local_side[selected])


def select_boundary_axis(mesh, axis, value, absolute_tolerance=1.0e-12):
    axis = int(axis)
    if axis < 0 or axis >= mesh.dimension:
        raise ValueError("boundary axis is out of range")
    value = float(value)
    return select_boundary_sides(
        mesh,
        lambda centroids, _normals: np.isclose(centroids[:, axis], value, rtol=0.0, atol=absolute_tolerance),
    )


def write_sideset(folder, mesh, sideset):
    validate_sideset(mesh, sideset)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_raw(folder / "parent.int32.raw", sideset.parent, np.int32)
    write_raw(folder / "lfi.int16.raw", sideset.local_side, np.int16)
    metadata = {
        "element_type": SIDE_ELEMENT_TYPE[mesh.element_type],
        "size": sideset.size,
        "parent": "parent.int32.raw",
        "lfi": "lfi.int16.raw",
        "rpath": True,
    }
    (folder / "meta.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return folder / "meta.yaml"


def read_sideset(folder):
    folder = Path(folder)
    metadata_path = folder / "meta.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"sideset metadata must be a mapping: {metadata_path}")
    parent = read_raw(folder / metadata.get("parent", "parent.raw"), dtype=np.int32)
    local_side = read_raw(folder / metadata.get("lfi", "lfi.int16.raw"), dtype=np.int16)
    sideset = Sideset(parent, local_side)
    if "size" in metadata and metadata["size"] != sideset.size:
        raise ValueError(f"sideset metadata size does not match streams in {folder}")
    return sideset


def write_nodeset(path, nodes):
    nodes = np.asarray(nodes, dtype=np.int64)
    if nodes.ndim != 1 or np.any(nodes < 0):
        raise ValueError("nodeset must be a one-dimensional array of non-negative indices")
    if len(nodes) and np.any(nodes[1:] <= nodes[:-1]):
        raise ValueError("nodeset must be sorted and unique")
    return write_raw(path, nodes, np.int32)


def read_nodeset(path):
    nodes = read_raw(path, dtype=np.int32).astype(np.int64)
    if len(nodes) and (np.any(nodes < 0) or np.any(nodes[1:] <= nodes[:-1])):
        raise ValueError("nodeset must be sorted, unique, and non-negative")
    return nodes
