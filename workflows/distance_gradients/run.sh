#!/usr/bin/env bash

set -e
set -x

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

SFEM_DISTANCE_GRADIENT_OUTPUT="${SFEM_DISTANCE_GRADIENT_OUTPUT:-distance_gradients}"
SFEM_DISTANCE_GRADIENT_STEPS="${SFEM_DISTANCE_GRADIENT_STEPS:-8}"

rm -rf "$SFEM_DISTANCE_GRADIENT_OUTPUT"

TEST_BIN="${SFEM_DISTANCE_GRADIENT_TEST:-}"
if [[ -z "$TEST_BIN" && -x "$ROOT/build64/sfem_DistanceGradientsTest" ]]; then
    TEST_BIN="$ROOT/build64/sfem_DistanceGradientsTest"
fi

if [[ -z "$TEST_BIN" && -x "$ROOT/build/sfem_DistanceGradientsTest" ]]; then
    TEST_BIN="$ROOT/build/sfem_DistanceGradientsTest"
fi

if [[ -z "$TEST_BIN" ]]; then
    TEST_BIN="$(command -v sfem_DistanceGradientsTest || true)"
fi

if [[ -z "$TEST_BIN" ]]; then
    echo "Could not find sfem_DistanceGradientsTest. Set SFEM_DISTANCE_GRADIENT_TEST=/path/to/sfem_DistanceGradientsTest." >&2
    exit 1
fi

SFEM_DISTANCE_GRADIENT_OUTPUT="$SFEM_DISTANCE_GRADIENT_OUTPUT" \
SFEM_DISTANCE_GRADIENT_STEPS="$SFEM_DISTANCE_GRADIENT_STEPS" \
    $LAUNCH "$TEST_BIN"

python3 - "$SFEM_DISTANCE_GRADIENT_OUTPUT" "$((SFEM_DISTANCE_GRADIENT_STEPS + 1))" <<'PY'
import glob
import os
import sys
import xml.etree.ElementTree as ET

out = os.path.abspath(sys.argv[1])
nsteps = int(sys.argv[2])
fields = os.path.join(out, "fields")

type_info = {
    "float32": ("Float", "4", 4),
    "float64": ("Float", "8", 8),
    "int32": ("Int", "4", 4),
    "int64": ("Int", "8", 8),
}

def one(pattern):
    matches = sorted(glob.glob(os.path.join(out, pattern)))
    if not matches:
        raise RuntimeError(f"missing file matching {pattern}")
    return matches[0]

def one_field(name, step):
    matches = sorted(glob.glob(os.path.join(fields, f"{name}.{step:06d}.*")))
    if not matches:
        raise RuntimeError(f"missing field {name} at step {step}")
    return matches[0]

def ext(path):
    return os.path.basename(path).rsplit(".", 1)[1]

def rel(path):
    return os.path.relpath(path, out)

def data_item(parent, path, dims):
    number_type, precision, _ = type_info[ext(path)]
    item = ET.SubElement(parent, "DataItem", {
        "Dimensions": dims,
        "NumberType": number_type,
        "Precision": precision,
        "Format": "Binary",
        "Endian": "Little" if sys.byteorder == "little" else "Big",
    })
    item.text = rel(path)
    return item

points = one("xdmf_points.*")
triangles = one("xdmf_triangles.*")
npoints = os.path.getsize(points) // (type_info[ext(points)][2] * 3)
ntriangles = os.path.getsize(triangles) // (type_info[ext(triangles)][2] * 3)

xdmf = ET.Element("Xdmf", {"Version": "3.0"})
domain = ET.SubElement(xdmf, "Domain")
surface_collection = ET.SubElement(domain, "Grid", {
    "Name": "distance_gradients",
    "GridType": "Collection",
    "CollectionType": "Temporal",
})

edge_xdmf = ET.Element("Xdmf", {"Version": "3.0"})
edge_domain = ET.SubElement(edge_xdmf, "Domain")
edge_collection = ET.SubElement(edge_domain, "Grid", {
    "Name": "edge_closest_point_gradients",
    "GridType": "Collection",
    "CollectionType": "Temporal",
})

edge_lines_xdmf = ET.Element("Xdmf", {"Version": "3.0"})
edge_lines_domain = ET.SubElement(edge_lines_xdmf, "Domain")
edge_lines_collection = ET.SubElement(edge_lines_domain, "Grid", {
    "Name": "edge_gradient_lines",
    "GridType": "Collection",
    "CollectionType": "Temporal",
})

attributes = [
    ("disp", "Vector", "disp_vec", f"{npoints} 3"),
    ("pt_distance", "Scalar", "pt_distance", str(npoints)),
    ("pt_grad", "Vector", "pt_grad_vec", f"{npoints} 3"),
    ("pt_closest", "Vector", "pt_closest_vec", f"{npoints} 3"),
    ("ee_distance", "Scalar", "ee_distance", str(npoints)),
]

for step in range(nsteps):
    grid = ET.SubElement(surface_collection, "Grid", {"Name": f"step_{step:06d}", "GridType": "Uniform"})
    ET.SubElement(grid, "Time", {"Value": str(step)})

    topology = ET.SubElement(grid, "Topology", {
        "TopologyType": "Triangle",
        "NumberOfElements": str(ntriangles),
    })
    data_item(topology, triangles, f"{ntriangles} 3")

    geometry = ET.SubElement(grid, "Geometry", {"GeometryType": "XYZ"})
    data_item(geometry, points, f"{npoints} 3")

    for attr_name, attr_type, file_name, dims in attributes:
        attr = ET.SubElement(grid, "Attribute", {
            "Name": attr_name,
            "AttributeType": attr_type,
            "Center": "Node",
        })
        data_item(attr, one_field(file_name, step), dims)

    edge_points = one_field("ee_closest_points", step)
    edge_grads = one_field("ee_closest_grad", step)
    edge_distances = one_field("ee_closest_distance", step)
    edge_indices = one_field("ee_closest_indices", step)
    n_edge_points = os.path.getsize(edge_points) // (type_info[ext(edge_points)][2] * 3)

    edge_grid = ET.SubElement(edge_collection, "Grid", {"Name": f"edge_step_{step:06d}", "GridType": "Uniform"})
    ET.SubElement(edge_grid, "Time", {"Value": str(step)})
    edge_topology = ET.SubElement(edge_grid, "Topology", {
        "TopologyType": "Polyvertex",
        "NumberOfElements": str(n_edge_points),
    })
    data_item(edge_topology, edge_indices, str(n_edge_points))
    edge_geometry = ET.SubElement(edge_grid, "Geometry", {"GeometryType": "XYZ"})
    data_item(edge_geometry, edge_points, f"{n_edge_points} 3")

    edge_attr = ET.SubElement(edge_grid, "Attribute", {
        "Name": "ee_grad",
        "AttributeType": "Vector",
        "Center": "Node",
    })
    data_item(edge_attr, edge_grads, f"{n_edge_points} 3")

    edge_distance_attr = ET.SubElement(edge_grid, "Attribute", {
        "Name": "ee_distance",
        "AttributeType": "Scalar",
        "Center": "Node",
    })
    data_item(edge_distance_attr, edge_distances, str(n_edge_points))

    edge_line_points = one_field("ee_gradient_line_points", step)
    edge_line_indices = one_field("ee_gradient_line_indices", step)
    edge_line_distances = one_field("ee_gradient_line_distance", step)
    n_edge_line_points = os.path.getsize(edge_line_points) // (type_info[ext(edge_line_points)][2] * 3)
    n_edge_lines = os.path.getsize(edge_line_indices) // (type_info[ext(edge_line_indices)][2] * 2)

    edge_line_grid = ET.SubElement(edge_lines_collection, "Grid", {"Name": f"edge_lines_step_{step:06d}", "GridType": "Uniform"})
    ET.SubElement(edge_line_grid, "Time", {"Value": str(step)})
    edge_line_topology = ET.SubElement(edge_line_grid, "Topology", {
        "TopologyType": "Polyline",
        "NumberOfElements": str(n_edge_lines),
        "NodesPerElement": "2",
    })
    data_item(edge_line_topology, edge_line_indices, f"{n_edge_lines} 2")
    edge_line_geometry = ET.SubElement(edge_line_grid, "Geometry", {"GeometryType": "XYZ"})
    data_item(edge_line_geometry, edge_line_points, f"{n_edge_line_points} 3")

    edge_line_distance_attr = ET.SubElement(edge_line_grid, "Attribute", {
        "Name": "ee_distance",
        "AttributeType": "Scalar",
        "Center": "Node",
    })
    data_item(edge_line_distance_attr, edge_line_distances, str(n_edge_line_points))

tree = ET.ElementTree(xdmf)
ET.indent(tree, space="  ")
tree.write(os.path.join(out, "distance_gradients.xdmf"), encoding="utf-8", xml_declaration=True)

edge_tree = ET.ElementTree(edge_xdmf)
ET.indent(edge_tree, space="  ")
edge_tree.write(os.path.join(out, "edge_gradients.xdmf"), encoding="utf-8", xml_declaration=True)

edge_lines_tree = ET.ElementTree(edge_lines_xdmf)
ET.indent(edge_lines_tree, space="  ")
edge_lines_tree.write(os.path.join(out, "edge_gradient_lines.xdmf"), encoding="utf-8", xml_declaration=True)
PY
