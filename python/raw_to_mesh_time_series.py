#!/usr/bin/env python3

import meshio
import numpy as np
import sys, getopt
import os
import glob

def main(argv):
    directory = "./"
    output_path = "out.xmf"
    field  = "data_*.raw"
    field_dtype = np.float32

    print('Open files with Paraview: Xdmf3ReaderT')

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
    # i3 = np.fromfile(f'{directory}/i3.raw', dtype=np.int32)

    nppoints = len(x)
    points = np.array([x, y, z]).transpose()
    cells = [
        # ("tetra", np.array([i0, i1, i2, i3]).transpose())
        ("triangle", np.array([i0, i1, i2]).transpose())
    ]

    with meshio.xdmf.TimeSeriesWriter(output_path) as writer:
        writer.write_points_cells(points, cells)

        if field: 
            files = glob.glob(field, recursive=False)
            files.sort()
            # print(files)
            t = 0.
            for f in files:
                data = np.fromfile(f, dtype=field_dtype)

                if(len(data) != len(x)):
                    print(f"Error: data lenght is different from number of nodes {len(data)} != {len(x)}")
                    exit(1)
                    
                writer.write_data(t, point_data={"data": data})
                t += 1
            
if __name__ == '__main__':
    main(sys.argv)


