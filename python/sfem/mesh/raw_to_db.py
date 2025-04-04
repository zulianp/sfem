#!/usr/bin/env python3

import meshio
import numpy as np
import sys, getopt
import os
import glob
import pdb

import inspect

# Get the current frame
frame = inspect.currentframe()

tet4_names = ("tetra", "tetrahedron", "tetra4", "tet4", "TET4")
hex8_names = ("hexahedron", "hex", "hex8", "HEX8")

quad4_names = ("quad", "quad4", "QUAD4")
tri3_names = ("tri", "tri3", "TRI3")

try:
    geom_t
except NameError:
    print("raw_to_db: self contained mode")
    geom_t = np.float32
    idx_t = np.int32

max_nodes_x_element = 10


# def ssquad4_to_standard(ssref, idx, points):
#     if ssref == 2:


def write_transient_data(
    output_path,
    points,
    cells,
    point_data,
    point_data_type,
    cell_data,
    cell_data_type,
    n_time_steps,
    time_whole,
    time_step_format,
):

    with meshio.xdmf.TimeSeriesWriter(output_path) as writer:
        writer.write_points_cells(points, cells)

        # steps = [None] * n_time_steps
        cell_data_steps = [None] * n_time_steps

        if cell_data:
            paths = cell_data.split(",")
            types = cell_data_type.split(",")

            for p in paths:
                cell_data_files = glob.glob(p, recursive=False)
                cell_data_files.sort()

                if len(cell_data_files) < n_time_steps:
                    print(
                        f"Invalid sequence length {len(cell_data_files)} for pattern {p}"
                    )

                for i in range(0, n_time_steps):
                    if not cell_data_steps[i]:
                        cell_data_steps[i] = []

                    cell_data_steps[i].append(cell_data_files[i])

        point_data_steps = [None] * n_time_steps

        if point_data:
            paths = point_data.split(",")
            types = point_data_type.split(",")

            for p in paths:
                point_data_files = glob.glob(p, recursive=False)
                point_data_files.sort()

                if len(point_data_files) < n_time_steps:
                    print(
                        f"Invalid sequence length {len(point_data_files)} for pattern {p}"
                    )

                for i in range(0, n_time_steps):
                    if not point_data_steps[i]:
                        point_data_steps[i] = []

                    point_data_steps[i].append(point_data_files[i])

        for t in range(0, n_time_steps):
            name_to_point_data = {}
            name_to_cell_data = {}

            cds = point_data_steps[t]

            has_point_data = False
            has_cell_data = False

            if cds:
                has_point_data = True
                for cd in cds:
                    data = np.fromfile(cd, dtype=point_data_type)
                    name = os.path.basename(cd)
                    name = os.path.splitext(os.path.splitext(name)[0])[0]
                    name = name.replace(".", "_")
                    name_to_point_data[name] = data

                    if len(data) != len(data):
                        frame = inspect.currentframe()
                        print(
                            f"Error in {__file__} at line {frame.f_lineno}:\n .... data length is different from number of nodes {len(data)} != {len(data)}"
                        )
                        exit(1)

            cds = cell_data_steps[t]

            if cds:
                has_cell_data = True
                for cd in cds:
                    data = np.fromfile(cd, dtype=cell_data_type)
                    name = os.path.basename(cd)
                    name = os.path.splitext(os.path.splitext(name)[0])[0]
                    name = name.replace(".", "_")
                    name_to_cell_data[name] = data

                    if len(data) != len(data):
                        frame = inspect.currentframe()
                        print(
                            f"Error in {__file__} at line {frame.f_lineno}:\n .... data length is different from number of nodes {len(data)} != {len(data)}"
                        )
                        exit(1)

            if has_point_data and not has_cell_data:
                writer.write_data(time_whole[t], point_data=name_to_point_data)
            elif not has_point_data and has_cell_data:
                writer.write_data(time_whole[t], cell_data=name_to_cell_data)
            elif has_point_data and has_cell_data:
                writer.write_data(
                    time_whole[t],
                    point_data=name_to_point_data,
                    cell_data=name_to_cell_data,
                )


def add_fields(field_data, field_data_type, storage, check_len):
    if field_data:
        paths = field_data.split(",")
        types = field_data_type.split(",")

        n_paths = len(paths)

        if n_paths != len(types):
            if len(types) == 1:
                # One for all is allowed
                types = [types[0]] * len(paths)
            else:
                print("Size mismatch between field_data and field_data_type\n")
                sys.exit(1)

        for i in range(0, n_paths):
            p = paths[i]
            t = np.dtype(types[i])

            files = glob.glob(p, recursive=False)
            files.sort()

            for f in files:
                data = np.fromfile(f, dtype=t)
                name = os.path.splitext(os.path.basename(f))[0]

                if len(data) != check_len:
                    frame = inspect.currentframe()
                    print(
                        f"Error in {__file__} at line {frame.f_lineno}:\n .... data length is different from number of nodes {len(data)} != {check_len}"
                    )

                    sys.exit(1)

                print(
                    f"field: {name}, min={np.min(data)}, max={np.max(data)}, sum={np.sum(data)} type={t}"
                )
                storage[name] = data


def raw_to_db(argv):
    usage = f"usage: {argv[0]} <input_folder> <output_mesh>"

    if len(argv) < 3:
        print(usage)
        sys.exit(1)

    raw_mesh_folder = argv[1]
    output_path = argv[2]
    raw_xyz_folder = raw_mesh_folder

    point_data = None
    point_data_type = "float64"

    cell_data = None
    cell_data_type = "float64"

    transient = False
    time_step_format = "%s.%d.%d.raw"
    n_time_steps = 1
    time_whole = []

    cell_type = None
    verbose = False
    ssref = 0

    try:
        opts, args = getopt.getopt(
            argv[3:],
            "p:d:c:t:hv",
            [
                "coords=",
                "point_data=",
                "point_data_type=",
                "cell_type=",
                "cell_data=",
                "cell_data_type=",
                "transient",
                "time_step_format",
                "n_time_steps=",
                "time_whole=",
                "time_whole_txt=",
                "help",
                "verbose",
                "ssref",
            ],
        )

    except getopt.GetoptError as err:
        print(err)
        print(usage)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(usage)
            sys.exit()
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-p", "--point_data"):
            point_data = arg
        elif opt in ("-d", "--point_data_type"):
            point_data_type = arg
        elif opt in ("-c", "--cell_data"):
            cell_data = arg
        elif opt in ("-t", "--cell_data_type"):
            cell_data_type = arg
        elif opt in ("--transient"):
            transient = True
        elif opt in ("--time_whole"):
            time_whole = np.fromfile(arg, dtype=np.float32)
        elif opt in ("--time_whole_txt"):
            time_whole = np.loadtxt(arg, dtype=np.float32)
        elif opt in ("--time_step_format"):
            time_step_format = arg
        elif opt in ("--n_time_steps"):
            n_time_steps = int(arg)
        elif opt in ("--cell_type"):
            cell_type = arg
        elif opt in ("--coords"):
            raw_xyz_folder = arg
            if verbose:
                print(f"Using coords={arg}")
        elif opt in ("--ssref"):
            ssref = int(arg)

    if transient:
        if len(time_whole) == 0:
            assert n_time_steps != 0
            time_whole = np.arange(0, n_time_steps)
        else:
            n_time_steps = len(time_whole)

    points = []
    for pfn in ["x.raw", "y.raw", "z.raw"]:
        path = f"{raw_xyz_folder}/{pfn}"
        if os.path.exists(path):
            if verbose:
                print(f"Reading {path}...")
            x = np.fromfile(path, dtype=geom_t)
            points.append(x)

    # Attempt format x0, x1, x2
    if len(points) == 0:
        for d in range(0, 3):
            path = f"{raw_xyz_folder}/x{d}.raw"
            if os.path.exists(path):
                if verbose:
                    print(f"Reading {path}...")
                x = np.fromfile(path, dtype=geom_t)
                points.append(x)

    idx = []
    for i in range(0, max_nodes_x_element):
        path = f"{raw_mesh_folder}/i{i}.raw"
        if os.path.exists(path):
            if verbose:
                print(f"Reading {path}...")
            ii = np.fromfile(path, dtype=idx_t)
            idx.append(ii)
        else:
            path = f"{raw_mesh_folder}/i{i}.int32.raw"
            if os.path.exists(path):
                if verbose:
                    print(f"Reading {path}...")
                ii = np.fromfile(path, dtype=idx_t)
                idx.append(ii)
            else:
                # No more indices to read!
                break

    if cell_type in quad4_names:
        cell_type = "quad"

    if cell_type in hex8_names:
        cell_type = "hexahedron"

    if cell_type in tet4_names:
        cell_type = "tetra"

    if cell_type in tri3_names:
        cell_type = "triangle"

    # Do I need to do that?
    # if ssref > 1:
    #     # Convert ssmesh to standard mesh or to high-order rep
    #     assert cell_type != None

    #     if cell_type == "quad":
    #         idx, points = ssquad4_to_standard(ssref, idx, points)
    #     elif cell_type == "hexahedron"
    #         # Implement me!
    #         assert False

    if cell_type == None:
        if len(idx) == 3:
            cell_type = "triangle"
        elif len(idx) == 6:
            cell_type = "triangle6"
        elif len(idx) == 4:
            if len(points) == 2:
                cell_type = "quad"
            else:
                cell_type = "tetra"
        elif len(idx) == 8:
            cell_type = "hexahedron"
        elif len(idx) == 10:
            cell_type = "tetra10"
        elif len(idx) == 2:
            cell_type = "line"
        elif len(idx) == 1:
            cell_type = "vertex"

    print(f"numnodes = {len(idx)} -> {cell_type}")
    n_points = len(points[0])
    n_cells = len(idx[0])

    if n_points == 0 or n_cells == 0:
        print(f"Warning empty database at {raw_mesh_folder}")
        return

    points = np.array(points).transpose()
    cells = [(cell_type, np.array(idx).transpose())]

    if transient:
        print("Transient mode!")

        write_transient_data(
            output_path,
            points,
            cells,
            point_data,
            point_data_type,
            cell_data,
            cell_data_type,
            n_time_steps,
            time_whole,
            time_step_format,
        )
    else:
        mesh = meshio.Mesh(points, cells)

        add_fields(point_data, point_data_type, mesh.point_data, n_points)
        add_fields(cell_data, cell_data_type, mesh.cell_data, n_cells)

        mesh.write(output_path)


# Example usage
# ./raw_to_db.py raw_db mesh_db.vtk --point_data='raw_db/point_data/*,raw_db/x.raw' --point_data_type='float64,float32'
if __name__ == "__main__":
    raw_to_db(sys.argv)
