#!/usr/bin/env python3

import meshio
import numpy as np
import sys, getopt
import os
import glob


# Example usage
# ./raw_to_db.py raw_db mesh_db.vtk --point_data='raw_db/point_data/*,raw_db/x.raw' --point_data_type='float64,float32'

geom_type = np.float32
idx_type = np.int32
max_nodes_x_element = 10


def write_transient_data(
    output_path,
    points, cells, 
    point_data, point_data_type,
    cell_data, cell_data_type, n_time_steps, time_whole, time_step_format):

    with meshio.xdmf.TimeSeriesWriter(output_path) as writer:
        writer.write_points_cells(points, cells)

        # steps = [None] * n_time_steps 
        cell_data_steps = [None] * n_time_steps

        if cell_data: 
            paths = cell_data.split(',')
            types = cell_data_type.split(',')

            for p in paths:
                cell_data_files = glob.glob(p, recursive=False)
                cell_data_files.sort()

                for i in range(0, n_time_steps):
                    # print(f'{i} -> {os.path.basename(cell_data_files[i])} -> {cell_data_files[i]}')

                    if not cell_data_steps[i]:
                        cell_data_steps[i] = []

                    cell_data_steps[i].append(cell_data_files[i])


        point_data_steps = [None] * n_time_steps

        if point_data: 
            paths = point_data.split(',')
            types = point_data_type.split(',')

            for p in paths:
                point_data_files = glob.glob(p, recursive=False)
                point_data_files.sort()

                if(len(point_data_files) < n_time_steps):
                    print(f"Invalid sequence length {len(point_data_files)} for pattern {p}")

                for i in range(0, n_time_steps):
                    # print(f'{i} -> {os.path.basename(point_data_files[i])} -> {point_data_files[i]}')

                    if not point_data_steps[i]:
                        point_data_steps[i] = []

                    point_data_steps[i].append(point_data_files[i])

        # print(point_data_steps)

        for t in range(0, n_time_steps):
            name_to_point_data = {}

            cds = point_data_steps[t];
           
            if cds:
                for cd in cds:
                    data = np.fromfile(cd, dtype=point_data_type)
                    name = os.path.basename(cd)
                    name = os.path.splitext(os.path.splitext(name)[0])[0]
                    name = name.replace('.', '_')
                    name_to_point_data[name] = data

                    if(len(data) != len(data)):
                        print(f"Error: data lenght is different from number of nodes {len(data)} != {len(data)}")
                        exit(1)

                    # print(f"field: {name}, path: {cd}, min={round(np.min(data), 3)}, max={round(np.max(data), 3)}, sum={round(np.sum(data), 3)}")

            writer.write_data(time_whole[t], point_data=name_to_point_data)
            

def add_fields(field_data, field_data_type, storage, check_len):
    if field_data: 
        paths = field_data.split(',')
        types = field_data_type.split(',')

        n_paths = len(paths)

        if(n_paths != len(types)):
            if(len(types) == 1):
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

                # name = os.path.basename(f).split('.')[0]
                name = os.path.splitext(os.path.basename(f))[0]

                if(len(data) != check_len):
                    print(f"Error: data lenght is different from number of nodes {len(data)} != {check_len}")
                    sys.exit(1)

                print(f"field: {name}, min={np.min(data)}, max={np.max(data)}, sum={np.sum(data)}")
                storage[name] = data

def main(argv):
    usage = f'usage: {argv[0]} <input_folder> <output_mesh>'

    if(len(argv) < 3):
        print(usage)
        sys.exit(1)

    raw_mesh_folder = argv[1]
    output_path = argv[2]
    
    point_data  = None
    point_data_type = "float64"

    cell_data  = None
    cell_data_type = "float64"

    transient = False
    time_step_format = "%s.%d.%d.raw"
    n_time_steps = 1
    time_whole=[]

    try:
        opts, args = getopt.getopt(
            argv[3:], "p:d:c:t:h",
            ["point_data=", "point_data_type=", "cell_data=", "cell_data_type=", "transient", "time_step_format", "n_time_steps=", "time_whole=", "help"])

    except getopt.GetoptError as err:
        print(err)
        print(usage)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage)
            sys.exit()
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
        elif opt in ("--time_step_format"):
            time_step_format = arg
        elif opt in ("--n_time_steps"):
            n_time_steps = int(arg)

    if transient:
        if len(time_whole) == 0:
            assert n_time_steps != 0
            time_whole = np.arange(0, n_time_steps)
        else:
            assert n_time_steps == 0 or n_time_steps == len(time_whole)
            n_time_steps = len(time_whole)

    points = []
    for pfn in ['x.raw', 'y.raw', 'z.raw']:
        path = f'{raw_mesh_folder}/{pfn}'
        if os.path.exists(path):
            x = np.fromfile(path, dtype=geom_type)
            points.append(x)
            
    idx = []
    for i in range(0, max_nodes_x_element):
        path = f'{raw_mesh_folder}/i{i}.raw'
        if os.path.exists(path):
            ii = np.fromfile(path, dtype=idx_type)
            idx.append(ii)
        else:
            # No more indices to read!
            break
    
    cell_type = None
    if len(idx) == 3:
        cell_type = "triangle"
    elif len(idx) == 6:
        cell_type = "triangle6"
    elif len(idx) == 4:
        cell_type = "tetra"
    elif len(idx) == 8:
        cell_type = "hexahedron"
    elif len(idx) == 10:
        cell_type = "tetra10"
    elif len(idx) == 2:
        cell_type = "line"


    print(f'numnodes = {len(idx)} -> {cell_type}')

    n_points = len(points[0])
    n_cells = len(idx[0])

    points = np.array(points).transpose()
    cells = [
        (cell_type, np.array(idx).transpose())
    ]

    if transient:
        print("Transient mode!")

        write_transient_data(
           output_path,
           points, cells, 
           point_data, point_data_type,
           cell_data, cell_data_type, n_time_steps, time_whole, time_step_format)
    else:
        mesh = meshio.Mesh(points, cells)

        add_fields(point_data, point_data_type, mesh.point_data, n_points)
        add_fields(cell_data, cell_data_type, mesh.cell_data, n_cells)

        mesh.write(output_path)

if __name__ == '__main__':
    main(sys.argv)

