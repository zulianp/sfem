"""Deterministic first-order meshes for verification and validation cases."""

import math

import numpy as np

from .mesh import Mesh


_HEX_TO_TET = np.asarray(
    [
        [0, 1, 3, 7],
        [0, 1, 7, 5],
        [0, 4, 5, 7],
        [1, 2, 3, 6],
        [1, 3, 7, 6],
        [1, 5, 6, 7],
    ],
    dtype=np.int64,
)


def _positive_float(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _element_type(value, supported):
    result = str(value).upper()
    if result not in supported:
        raise ValueError(f"unsupported element type {value}; expected one of {', '.join(supported)}")
    return result


def _orient_tets(points, elements):
    elements = np.asarray(elements, dtype=np.int64).copy()
    coordinates = points[elements]
    determinants = np.linalg.det(coordinates[:, 1:] - coordinates[:, :1])
    if np.any(np.abs(determinants) <= np.finfo(np.float64).eps):
        raise ValueError("generated a degenerate tetrahedron")
    flipped = determinants < 0
    temporary = elements[flipped, 1].copy()
    elements[flipped, 1] = elements[flipped, 2]
    elements[flipped, 2] = temporary
    return elements


def _triangulate_quads(quads):
    triangles = np.empty((2 * len(quads), 3), dtype=np.int64)
    triangles[0::2] = quads[:, (0, 1, 3)]
    triangles[1::2] = quads[:, (1, 2, 3)]
    return triangles


def _split_hexes(points, hexes):
    tets = np.asarray(hexes, dtype=np.int64)[:, _HEX_TO_TET].reshape(-1, 4)
    return _orient_tets(points, tets)


def rectangle_mesh(width, height, nx, ny, element_type="QUAD4", origin=(0.0, 0.0)):
    width = _positive_float(width, "width")
    height = _positive_float(height, "height")
    nx = _positive_int(nx, "nx")
    ny = _positive_int(ny, "ny")
    element_type = _element_type(element_type, ("QUAD4", "TRI3"))
    origin = np.asarray(origin, dtype=np.float64)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin must contain two finite coordinates")

    x = np.linspace(origin[0], origin[0] + width, nx + 1)
    y = np.linspace(origin[1], origin[1] + height, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))

    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    n00 = (iy * (nx + 1) + ix).ravel()
    quads = np.column_stack((n00, n00 + 1, n00 + nx + 2, n00 + nx + 1))
    elements = quads if element_type == "QUAD4" else _triangulate_quads(quads)
    return Mesh(points, elements, element_type)


def box_mesh(width, height, depth, nx, ny, nz, element_type="HEX8", origin=(0.0, 0.0, 0.0)):
    width = _positive_float(width, "width")
    height = _positive_float(height, "height")
    depth = _positive_float(depth, "depth")
    nx = _positive_int(nx, "nx")
    ny = _positive_int(ny, "ny")
    nz = _positive_int(nz, "nz")
    element_type = _element_type(element_type, ("HEX8", "TET4"))
    origin = np.asarray(origin, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin must contain three finite coordinates")

    x = np.linspace(origin[0], origin[0] + width, nx + 1)
    y = np.linspace(origin[1], origin[1] + height, ny + 1)
    z = np.linspace(origin[2], origin[2] + depth, nz + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    def node(i, j, k):
        return k + (nz + 1) * (i + (nx + 1) * j)

    iz, ix, iy = np.meshgrid(np.arange(nz), np.arange(nx), np.arange(ny), indexing="ij")
    ix = ix.ravel()
    iy = iy.ravel()
    iz = iz.ravel()
    hexes = np.column_stack(
        (
            node(ix, iy, iz),
            node(ix + 1, iy, iz),
            node(ix + 1, iy + 1, iz),
            node(ix, iy + 1, iz),
            node(ix, iy, iz + 1),
            node(ix + 1, iy, iz + 1),
            node(ix + 1, iy + 1, iz + 1),
            node(ix, iy + 1, iz + 1),
        )
    )
    elements = hexes if element_type == "HEX8" else _split_hexes(points, hexes)
    return Mesh(points, elements, element_type)


def promote_simplex_mesh(mesh, element_type):
    """Promote a TRI3 or TET4 mesh using globally shared edge midpoints."""

    element_type = str(element_type).upper()
    if mesh.element_type == "TRI3" and element_type == "TRI6":
        local_edges = ((0, 1), (1, 2), (0, 2))
    elif mesh.element_type == "TET4" and element_type == "TET10":
        local_edges = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
    else:
        raise ValueError(f"cannot promote {mesh.element_type} to {element_type}")

    edge_nodes = {}
    points = [point.copy() for point in mesh.points]
    promoted = np.empty((mesh.n_elements, len(mesh.elements[0]) + len(local_edges)), dtype=np.int64)
    promoted[:, : mesh.elements.shape[1]] = mesh.elements
    for element_index, element in enumerate(mesh.elements):
        for edge_index, (left, right) in enumerate(local_edges):
            edge = tuple(sorted((int(element[left]), int(element[right]))))
            node = edge_nodes.get(edge)
            if node is None:
                node = len(points)
                edge_nodes[edge] = node
                points.append(0.5 * (mesh.points[edge[0]] + mesh.points[edge[1]]))
            promoted[element_index, mesh.elements.shape[1] + edge_index] = node
    return Mesh(np.asarray(points), promoted, element_type)


def tensor_product_mesh(width, height, depth, nx, ny, nz, element_type, origin=(0.0, 0.0, 0.0)):
    """Generate SFEM's standard or lexicographic tensor-product element layouts."""

    element_type = _element_type(element_type, ("HEX27", "PROTEUS_HEX8", "PROTEUS_HEX27"))
    if element_type == "PROTEUS_HEX8":
        mesh = box_mesh(width, height, depth, nx, ny, nz, "HEX8", origin)
        return Mesh(mesh.points, mesh.elements[:, (0, 1, 3, 2, 4, 5, 7, 6)], element_type)

    width = _positive_float(width, "width")
    height = _positive_float(height, "height")
    depth = _positive_float(depth, "depth")
    nx = _positive_int(nx, "nx")
    ny = _positive_int(ny, "ny")
    nz = _positive_int(nz, "nz")
    origin = np.asarray(origin, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin must contain three finite coordinates")

    nx2, ny2, nz2 = 2 * nx, 2 * ny, 2 * nz
    x = np.linspace(origin[0], origin[0] + width, nx2 + 1)
    y = np.linspace(origin[1], origin[1] + height, ny2 + 1)
    z = np.linspace(origin[2], origin[2] + depth, nz2 + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    def node(i, j, k):
        return k + (nz2 + 1) * (i + (nx2 + 1) * j)

    cartesian = []
    for j in range(ny):
        for i in range(nx):
            for k in range(nz):
                cartesian.append(
                    [node(2 * i + dx, 2 * j + dy, 2 * k + dz)
                     for dz in range(3) for dy in range(3) for dx in range(3)]
                )
    cartesian = np.asarray(cartesian, dtype=np.int64)
    if element_type == "PROTEUS_HEX27":
        elements = cartesian
    else:
        standard_order = (0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23, 25, 21,
                          9, 11, 17, 15, 10, 14, 16, 12, 4, 22, 13)
        elements = cartesian[:, standard_order]
    return Mesh(points, elements, element_type)


def proteus_rectangle_mesh(width, height, nx, ny, origin=(0.0, 0.0)):
    mesh = rectangle_mesh(width, height, nx, ny, "QUAD4", origin)
    return Mesh(mesh.points, mesh.elements[:, (0, 1, 3, 2)], "PROTEUS_QUAD4")


def _polar_mesh(inner_radius, outer_radius, radial_cells, angular_cells, angle_start, angle_end, periodic,
                element_type):
    inner_radius = _positive_float(inner_radius, "inner_radius")
    outer_radius = _positive_float(outer_radius, "outer_radius")
    if outer_radius <= inner_radius:
        raise ValueError("outer_radius must be larger than inner_radius")
    radial_cells = _positive_int(radial_cells, "radial_cells")
    angular_cells = _positive_int(angular_cells, "angular_cells")
    if periodic and angular_cells < 3:
        raise ValueError("a periodic annulus requires at least three angular cells")
    element_type = _element_type(element_type, ("QUAD4", "TRI3"))
    angle_start = float(angle_start)
    angle_end = float(angle_end)
    if not math.isfinite(angle_start) or not math.isfinite(angle_end) or angle_end <= angle_start:
        raise ValueError("angle_end must be finite and larger than angle_start")

    radii = np.linspace(inner_radius, outer_radius, radial_cells + 1)
    angles = np.linspace(angle_start, angle_end, angular_cells, endpoint=False) if periodic else np.linspace(
        angle_start, angle_end, angular_cells + 1
    )
    rr, theta = np.meshgrid(radii, angles, indexing="xy")
    points = np.column_stack((rr.ravel() * np.cos(theta.ravel()), rr.ravel() * np.sin(theta.ravel())))
    angular_nodes = angular_cells if periodic else angular_cells + 1

    angle_index, radial_index = np.meshgrid(np.arange(angular_cells), np.arange(radial_cells), indexing="ij")
    angle_index = angle_index.ravel()
    radial_index = radial_index.ravel()
    next_angle = (angle_index + 1) % angular_nodes
    n00 = angle_index * (radial_cells + 1) + radial_index
    n01 = next_angle * (radial_cells + 1) + radial_index
    quads = np.column_stack((n00, n00 + 1, n01 + 1, n01))
    elements = quads if element_type == "QUAD4" else _triangulate_quads(quads)
    return Mesh(points, elements, element_type)


def annulus_mesh(inner_radius, outer_radius, radial_cells, angular_cells, element_type="QUAD4"):
    return _polar_mesh(
        inner_radius,
        outer_radius,
        radial_cells,
        angular_cells,
        0.0,
        2.0 * math.pi,
        True,
        element_type,
    )


def annular_sector_mesh(inner_radius, outer_radius, radial_cells, angular_cells, angle_start=0.0,
                        angle_end=0.5 * math.pi, element_type="QUAD4"):
    return _polar_mesh(
        inner_radius,
        outer_radius,
        radial_cells,
        angular_cells,
        angle_start,
        angle_end,
        False,
        element_type,
    )


def cylindrical_sector_mesh(inner_radius, outer_radius, length, radial_cells, angular_cells, axial_cells,
                            angle_start=0.0, angle_end=0.5 * math.pi, element_type="HEX8"):
    inner_radius = _positive_float(inner_radius, "inner_radius")
    outer_radius = _positive_float(outer_radius, "outer_radius")
    length = _positive_float(length, "length")
    if outer_radius <= inner_radius:
        raise ValueError("outer_radius must be larger than inner_radius")
    radial_cells = _positive_int(radial_cells, "radial_cells")
    angular_cells = _positive_int(angular_cells, "angular_cells")
    axial_cells = _positive_int(axial_cells, "axial_cells")
    element_type = _element_type(element_type, ("HEX8", "TET4"))
    if not math.isfinite(angle_start) or not math.isfinite(angle_end) or angle_end <= angle_start:
        raise ValueError("angle_end must be finite and larger than angle_start")

    radii = np.linspace(inner_radius, outer_radius, radial_cells + 1)
    angles = np.linspace(angle_start, angle_end, angular_cells + 1)
    axial = np.linspace(0.0, length, axial_cells + 1)
    rr, theta, zz = np.meshgrid(radii, angles, axial, indexing="ij")
    points = np.column_stack((rr.ravel() * np.cos(theta.ravel()), rr.ravel() * np.sin(theta.ravel()), zz.ravel()))

    def node(i, j, k):
        return k + (axial_cells + 1) * (j + (angular_cells + 1) * i)

    ir, it, iz = np.meshgrid(
        np.arange(radial_cells), np.arange(angular_cells), np.arange(axial_cells), indexing="ij"
    )
    ir = ir.ravel()
    it = it.ravel()
    iz = iz.ravel()
    hexes = np.column_stack(
        (
            node(ir, it, iz),
            node(ir + 1, it, iz),
            node(ir + 1, it + 1, iz),
            node(ir, it + 1, iz),
            node(ir, it, iz + 1),
            node(ir + 1, it, iz + 1),
            node(ir + 1, it + 1, iz + 1),
            node(ir, it + 1, iz + 1),
        )
    )
    elements = hexes if element_type == "HEX8" else _split_hexes(points, hexes)
    return Mesh(points, elements, element_type)


def _octahedron_surface(frequency):
    axes = np.eye(3)
    faces = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                face = np.asarray((sx * axes[0], sy * axes[1], sz * axes[2]))
                if np.dot(np.cross(face[1] - face[0], face[2] - face[0]), np.sum(face, axis=0)) < 0:
                    face[[1, 2]] = face[[2, 1]]
                faces.append(face)

    points = []
    point_ids = {}

    def add_point(point):
        direction = point / np.linalg.norm(point)
        key = tuple(np.round(direction, 14))
        if key not in point_ids:
            point_ids[key] = len(points)
            points.append(direction)
        return point_ids[key]

    triangles = []
    for face in faces:
        local = {}
        for i in range(frequency + 1):
            for j in range(frequency + 1 - i):
                point = ((frequency - i - j) * face[0] + i * face[1] + j * face[2]) / frequency
                local[i, j] = add_point(point)
        for i in range(frequency):
            for j in range(frequency - i):
                triangles.append((local[i, j], local[i + 1, j], local[i, j + 1]))
                if i + j <= frequency - 2:
                    triangles.append((local[i + 1, j], local[i + 1, j + 1], local[i, j + 1]))

    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    coordinates = points[triangles]
    inward = np.einsum(
        "ij,ij->i",
        np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]),
        np.sum(coordinates, axis=1),
    ) < 0
    temporary = triangles[inward, 1].copy()
    triangles[inward, 1] = triangles[inward, 2]
    triangles[inward, 2] = temporary
    return points, triangles


def spherical_shell_mesh(inner_radius, outer_radius, radial_cells, surface_frequency=1):
    inner_radius = _positive_float(inner_radius, "inner_radius")
    outer_radius = _positive_float(outer_radius, "outer_radius")
    if outer_radius <= inner_radius:
        raise ValueError("outer_radius must be larger than inner_radius")
    radial_cells = _positive_int(radial_cells, "radial_cells")
    surface_frequency = _positive_int(surface_frequency, "surface_frequency")

    directions, surface_triangles = _octahedron_surface(surface_frequency)
    radii = np.linspace(inner_radius, outer_radius, radial_cells + 1)
    points = np.concatenate([radius * directions for radius in radii], axis=0)
    layer_size = len(directions)
    tetrahedra = []
    for layer in range(radial_cells):
        lower_offset = layer * layer_size
        upper_offset = (layer + 1) * layer_size
        for triangle in surface_triangles:
            a, b, c = np.sort(triangle)
            lower = np.asarray((a, b, c), dtype=np.int64) + lower_offset
            upper = np.asarray((a, b, c), dtype=np.int64) + upper_offset
            tetrahedra.extend(
                (
                    (lower[0], lower[1], lower[2], upper[2]),
                    (lower[0], lower[1], upper[1], upper[2]),
                    (lower[0], upper[0], upper[1], upper[2]),
                )
            )
    elements = _orient_tets(points, np.asarray(tetrahedra, dtype=np.int64))
    return Mesh(points, elements, "TET4")


def spherical_shell_octant_mesh(inner_radius, outer_radius, radial_cells, surface_frequency, element_type="TET4"):
    """One spherical-shell octant, triangulated or assembled from cubed-sphere patches."""
    inner_radius = _positive_float(inner_radius, "inner_radius")
    outer_radius = _positive_float(outer_radius, "outer_radius")
    if outer_radius <= inner_radius:
        raise ValueError("outer_radius must be larger than inner_radius")
    radial_cells = _positive_int(radial_cells, "radial_cells")
    surface_frequency = _positive_int(surface_frequency, "surface_frequency")
    element_type = _element_type(element_type, ("TET4", "HEX8"))

    if element_type == "TET4":
        full_directions, full_triangles = _octahedron_surface(surface_frequency)
        in_octant = np.all(full_directions >= -1.0e-13, axis=1)
        retained = np.all(in_octant[full_triangles], axis=1)
        triangles = full_triangles[retained]
        used = np.unique(triangles.ravel())
        old_to_new = np.full(len(full_directions), -1, dtype=np.int64)
        old_to_new[used] = np.arange(len(used))
        directions = full_directions[used]
        triangles = old_to_new[triangles]
        radii = np.linspace(inner_radius, outer_radius, radial_cells + 1)
        points = np.concatenate([radius * directions for radius in radii], axis=0)
        layer_size = len(directions)
        lower_triangles = np.sort(triangles, axis=1)
        tetrahedra = np.empty((radial_cells * len(triangles) * 3, 4), dtype=np.int64)
        for layer in range(radial_cells):
            lower = lower_triangles + layer * layer_size
            upper = lower + layer_size
            block = tetrahedra[layer * len(triangles) * 3:(layer + 1) * len(triangles) * 3]
            block[0::3] = np.column_stack((lower, upper[:, 2]))
            block[1::3] = np.column_stack((lower[:, 0], lower[:, 1], upper[:, 1], upper[:, 2]))
            block[2::3] = np.column_stack((lower[:, 0], upper))
        return Mesh(points, _orient_tets(points, tetrahedra), "TET4")

    frequency = surface_frequency
    direction_ids = {}
    directions = []

    def add_direction(vector):
        direction = vector / np.linalg.norm(vector)
        key = tuple(np.round(direction, 14))
        if key not in direction_ids:
            direction_ids[key] = len(directions)
            directions.append(direction)
        return direction_ids[key]

    surface_quads = []
    for face in range(3):
        ids = np.empty((frequency + 1, frequency + 1), dtype=np.int64)
        for i in range(frequency + 1):
            for j in range(frequency + 1):
                vector = np.asarray((i / frequency, j / frequency, 1.0))
                vector = np.roll(vector, face + 1)
                ids[i, j] = add_direction(vector)
        for i in range(frequency):
            for j in range(frequency):
                surface_quads.append((ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]))

    directions = np.asarray(directions)
    radii = np.linspace(inner_radius, outer_radius, radial_cells + 1)
    points = np.concatenate([radius * directions for radius in radii], axis=0)
    layer_size = len(directions)
    quads = np.asarray(surface_quads, dtype=np.int64)
    hexes = np.concatenate(
        [
            np.column_stack((quads + layer * layer_size, quads + (layer + 1) * layer_size))
            for layer in range(radial_cells)
        ],
        axis=0,
    )
    return Mesh(points, hexes, "HEX8")
