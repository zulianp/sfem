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
max_nodes_x_element = 8

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

                name = os.path.basename(f).split('.')[0]

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

    try:
        opts, args = getopt.getopt(
            argv[3:], "p:d:c:t:h",
            ["point_data=", "point_data_type=", "cell_data=", "cell_data_type=", "help"])

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

    if len(idx) == 4:
        cell_type = "tetra"
    elif len(idx) == 8:
        cell_type = "hexahedron"

    n_points = len(points[0])
    n_cells = len(idx[0])

    points = np.array(points).transpose()
    cells = [
        (cell_type, np.array(idx).transpose())
    ]

    mesh = meshio.Mesh(points, cells)

    add_fields(point_data, point_data_type, mesh.point_data, n_points)
    add_fields(cell_data, cell_data_type, mesh.cell_data, n_cells)

    mesh.write(output_path)

if __name__ == '__main__':
    main(sys.argv)

