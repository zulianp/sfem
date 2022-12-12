#!/usr/bin/env python3

import meshio
import numpy as np
import sys, getopt

def main(argv):
    directory = "./"
    output_path = "out.vtu"
    field  = None
    field_dtype = np.float64

    try:
        opts, args = getopt.getopt(
            argv[1:], "d:f:o:h",
            ["dir=", "field=", "field_dtype=", "output=", "help"])

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(f'{argv[0]} -d <directory of raw mesh>')
            sys.exit()
        elif opt in ("-d", "--dir"):
            directory = arg
        elif opt in ("-f", "--field"):
            field = arg
        elif opt in ("-o", "--output"):
            output_path = arg
        elif opt in ("--field_dtype"):
            field_dtype = np.dtype(arg)

    x = np.fromfile(f'{directory}/x.raw', dtype=np.float32)
    y = np.fromfile(f'{directory}/y.raw', dtype=np.float32)
    z = np.fromfile(f'{directory}/z.raw', dtype=np.float32)

    i0 = np.fromfile(f'{directory}/i0.raw', dtype=np.int32)
    i1 = np.fromfile(f'{directory}/i1.raw', dtype=np.int32)
    i2 = np.fromfile(f'{directory}/i2.raw', dtype=np.int32)
    i3 = np.fromfile(f'{directory}/i3.raw', dtype=np.int32)

    nppoints = len(x)
    points = np.array([x, y, z]).transpose()
    cells = [
        ("tetra", np.array([i0, i1, i2, i3]).transpose())
    ]

    mesh = meshio.Mesh(points, cells)

    if field: 
        data = np.fromfile(field, dtype=field_dtype)
        # data = np.fromfile(field, dtype=np.float32)
            
        print(f'min={np.min(data)}')
        print(f'max={np.max(data)}')
        print(f'sum={np.sum(data)}')
        
        mesh.point_data["X"] = data

    mesh.write(output_path)

if __name__ == '__main__':
    main(sys.argv)

