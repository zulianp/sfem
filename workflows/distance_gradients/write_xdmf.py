#!/usr/bin/env python3

import glob
import os
import sys
import xml.etree.ElementTree as ET


TYPE_INFO = {
    "float32": ("Float", "4", 4),
    "float64": ("Float", "8", 8),
    "int32": ("Int", "4", 4),
    "int64": ("Int", "8", 8),
}


def extension(path):
    return os.path.basename(path).rsplit(".", 1)[1]


def relative_to_output(output_dir, path):
    return os.path.relpath(path, output_dir)


def find_one(output_dir, pattern):
    matches = sorted(glob.glob(os.path.join(output_dir, pattern)))
    if not matches:
        raise RuntimeError(f"missing file matching {pattern}")
    return matches[0]


def find_field(fields_dir, name, step):
    matches = sorted(glob.glob(os.path.join(fields_dir, f"{name}.{step:06d}.*")))
    if not matches:
        raise RuntimeError(f"missing field {name} at step {step}")
    return matches[0]


def add_data_item(output_dir, parent, path, dims):
    number_type, precision, _ = TYPE_INFO[extension(path)]
    item = ET.SubElement(
        parent,
        "DataItem",
        {
            "Dimensions": dims,
            "NumberType": number_type,
            "Precision": precision,
            "Format": "Binary",
            "Endian": "Little" if sys.byteorder == "little" else "Big",
        },
    )
    item.text = relative_to_output(output_dir, path)
    return item


def add_surface_step(output_dir, fields_dir, collection, step, points, triangles, npoints, ntriangles):
    grid = ET.SubElement(collection, "Grid", {"Name": f"step_{step:06d}", "GridType": "Uniform"})
    ET.SubElement(grid, "Time", {"Value": str(step)})

    topology = ET.SubElement(
        grid,
        "Topology",
        {
            "TopologyType": "Triangle",
            "NumberOfElements": str(ntriangles),
        },
    )
    add_data_item(output_dir, topology, triangles, f"{ntriangles} 3")

    geometry = ET.SubElement(grid, "Geometry", {"GeometryType": "XYZ"})
    add_data_item(output_dir, geometry, points, f"{npoints} 3")

    attributes = [
        ("disp", "Vector", "disp_vec", f"{npoints} 3"),
        ("pt_distance", "Scalar", "pt_distance", str(npoints)),
        ("pt_grad", "Vector", "pt_grad_vec", f"{npoints} 3"),
        ("pt_closest", "Vector", "pt_closest_vec", f"{npoints} 3"),
        ("ee_distance", "Scalar", "ee_distance", str(npoints)),
    ]

    for attr_name, attr_type, file_name, dims in attributes:
        attr = ET.SubElement(
            grid,
            "Attribute",
            {
                "Name": attr_name,
                "AttributeType": attr_type,
                "Center": "Node",
            },
        )
        add_data_item(output_dir, attr, find_field(fields_dir, file_name, step), dims)


def add_edge_point_step(output_dir, fields_dir, collection, step):
    edge_points = find_field(fields_dir, "ee_closest_points", step)
    edge_grads = find_field(fields_dir, "ee_closest_grad", step)
    edge_distances = find_field(fields_dir, "ee_closest_distance", step)
    edge_indices = find_field(fields_dir, "ee_closest_indices", step)
    n_edge_points = os.path.getsize(edge_points) // (TYPE_INFO[extension(edge_points)][2] * 3)

    grid = ET.SubElement(collection, "Grid", {"Name": f"edge_step_{step:06d}", "GridType": "Uniform"})
    ET.SubElement(grid, "Time", {"Value": str(step)})

    topology = ET.SubElement(
        grid,
        "Topology",
        {
            "TopologyType": "Polyvertex",
            "NumberOfElements": str(n_edge_points),
        },
    )
    add_data_item(output_dir, topology, edge_indices, str(n_edge_points))

    geometry = ET.SubElement(grid, "Geometry", {"GeometryType": "XYZ"})
    add_data_item(output_dir, geometry, edge_points, f"{n_edge_points} 3")

    grad_attr = ET.SubElement(
        grid,
        "Attribute",
        {
            "Name": "ee_grad",
            "AttributeType": "Vector",
            "Center": "Node",
        },
    )
    add_data_item(output_dir, grad_attr, edge_grads, f"{n_edge_points} 3")

    distance_attr = ET.SubElement(
        grid,
        "Attribute",
        {
            "Name": "ee_distance",
            "AttributeType": "Scalar",
            "Center": "Node",
        },
    )
    add_data_item(output_dir, distance_attr, edge_distances, str(n_edge_points))


def add_edge_line_step(output_dir, fields_dir, collection, step):
    points = find_field(fields_dir, "ee_gradient_line_points", step)
    indices = find_field(fields_dir, "ee_gradient_line_indices", step)
    distances = find_field(fields_dir, "ee_gradient_line_distance", step)
    n_points = os.path.getsize(points) // (TYPE_INFO[extension(points)][2] * 3)
    n_lines = os.path.getsize(indices) // (TYPE_INFO[extension(indices)][2] * 2)

    grid = ET.SubElement(collection, "Grid", {"Name": f"edge_lines_step_{step:06d}", "GridType": "Uniform"})
    ET.SubElement(grid, "Time", {"Value": str(step)})

    topology = ET.SubElement(
        grid,
        "Topology",
        {
            "TopologyType": "Polyline",
            "NumberOfElements": str(n_lines),
            "NodesPerElement": "2",
        },
    )
    add_data_item(output_dir, topology, indices, f"{n_lines} 2")

    geometry = ET.SubElement(grid, "Geometry", {"GeometryType": "XYZ"})
    add_data_item(output_dir, geometry, points, f"{n_points} 3")

    distance_attr = ET.SubElement(
        grid,
        "Attribute",
        {
            "Name": "ee_distance",
            "AttributeType": "Scalar",
            "Center": "Node",
        },
    )
    add_data_item(output_dir, distance_attr, distances, str(n_points))


def add_bounding_box_step(output_dir, fields_dir, collection, step):
    points = find_field(fields_dir, "bbox_points", step)
    indices = find_field(fields_dir, "bbox_indices", step)
    kinds = find_field(fields_dir, "bbox_kind", step)
    ids = find_field(fields_dir, "bbox_id", step)
    n_points = os.path.getsize(points) // (TYPE_INFO[extension(points)][2] * 3)
    n_lines = os.path.getsize(indices) // (TYPE_INFO[extension(indices)][2] * 2)

    grid = ET.SubElement(collection, "Grid", {"Name": f"bbox_step_{step:06d}", "GridType": "Uniform"})
    ET.SubElement(grid, "Time", {"Value": str(step)})

    topology = ET.SubElement(
        grid,
        "Topology",
        {
            "TopologyType": "Polyline",
            "NumberOfElements": str(n_lines),
            "NodesPerElement": "2",
        },
    )
    add_data_item(output_dir, topology, indices, f"{n_lines} 2")

    geometry = ET.SubElement(grid, "Geometry", {"GeometryType": "XYZ"})
    add_data_item(output_dir, geometry, points, f"{n_points} 3")

    kind_attr = ET.SubElement(
        grid,
        "Attribute",
        {
            "Name": "bbox_kind",
            "AttributeType": "Scalar",
            "Center": "Cell",
        },
    )
    add_data_item(output_dir, kind_attr, kinds, str(n_lines))

    id_attr = ET.SubElement(
        grid,
        "Attribute",
        {
            "Name": "bbox_id",
            "AttributeType": "Scalar",
            "Center": "Cell",
        },
    )
    add_data_item(output_dir, id_attr, ids, str(n_lines))


def write_xml(path, root):
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main(argv):
    if len(argv) != 3:
        raise SystemExit(f"usage: {argv[0]} <output_dir> <nsteps>")

    output_dir = os.path.abspath(argv[1])
    nsteps = int(argv[2])
    fields_dir = os.path.join(output_dir, "fields")

    points = find_one(output_dir, "xdmf_points.*")
    triangles = find_one(output_dir, "xdmf_triangles.*")
    npoints = os.path.getsize(points) // (TYPE_INFO[extension(points)][2] * 3)
    ntriangles = os.path.getsize(triangles) // (TYPE_INFO[extension(triangles)][2] * 3)

    surface_xdmf = ET.Element("Xdmf", {"Version": "3.0"})
    surface_domain = ET.SubElement(surface_xdmf, "Domain")
    surface_collection = ET.SubElement(
        surface_domain,
        "Grid",
        {
            "Name": "distance_gradients",
            "GridType": "Collection",
            "CollectionType": "Temporal",
        },
    )

    edge_xdmf = ET.Element("Xdmf", {"Version": "3.0"})
    edge_domain = ET.SubElement(edge_xdmf, "Domain")
    edge_collection = ET.SubElement(
        edge_domain,
        "Grid",
        {
            "Name": "edge_closest_point_gradients",
            "GridType": "Collection",
            "CollectionType": "Temporal",
        },
    )

    edge_lines_xdmf = ET.Element("Xdmf", {"Version": "3.0"})
    edge_lines_domain = ET.SubElement(edge_lines_xdmf, "Domain")
    edge_lines_collection = ET.SubElement(
        edge_lines_domain,
        "Grid",
        {
            "Name": "edge_gradient_lines",
            "GridType": "Collection",
            "CollectionType": "Temporal",
        },
    )

    bbox_xdmf = ET.Element("Xdmf", {"Version": "3.0"})
    bbox_domain = ET.SubElement(bbox_xdmf, "Domain")
    bbox_collection = ET.SubElement(
        bbox_domain,
        "Grid",
        {
            "Name": "bounding_boxes",
            "GridType": "Collection",
            "CollectionType": "Temporal",
        },
    )

    for step in range(nsteps):
        add_surface_step(output_dir, fields_dir, surface_collection, step, points, triangles, npoints, ntriangles)
        add_edge_point_step(output_dir, fields_dir, edge_collection, step)
        add_edge_line_step(output_dir, fields_dir, edge_lines_collection, step)
        add_bounding_box_step(output_dir, fields_dir, bbox_collection, step)

    write_xml(os.path.join(output_dir, "distance_gradients.xdmf"), surface_xdmf)
    write_xml(os.path.join(output_dir, "edge_gradients.xdmf"), edge_xdmf)
    write_xml(os.path.join(output_dir, "edge_gradient_lines.xdmf"), edge_lines_xdmf)
    write_xml(os.path.join(output_dir, "bounding_boxes.xdmf"), bbox_xdmf)


if __name__ == "__main__":
    main(sys.argv)
