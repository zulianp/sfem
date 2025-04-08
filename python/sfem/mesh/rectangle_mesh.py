#!/usr/bin/env python3

import numpy as np
import sys, getopt
import os
import glob

tri3_names = ("triangle", "tri", "tri3", "TRI3")
quad4_names = ("quadrilateral", "quad", "quad4", "QUAD4")

idx_t = np.int32
geom_t = np.float32


def to_sfem_element_type(elem_type):
    if elem_type in tri3_names:
        return "TRI3"
    elif elem_type in quad4_names:
        return "QUAD4"
    else:
        print(f"[Error] Unknow element_type: {elem_type}")
        exit(1)


def create(w, h, nx, ny, cell_type):
    gx = np.linspace(0, w, num=nx, dtype=geom_t)
    gy = np.linspace(0, h, num=ny, dtype=geom_t)

    x, y = np.meshgrid(gx, gy)

    x = np.reshape(x, x.shape[0] * x.shape[1])
    y = np.reshape(y, y.shape[0] * y.shape[1])

    points = [x, y]
    idx = []

    if cell_type in quad4_names:
        i0 = np.zeros((nx - 1) * (ny - 1), dtype=idx_t)
        i1 = np.zeros((nx - 1) * (ny - 1), dtype=idx_t)
        i2 = np.zeros((nx - 1) * (ny - 1), dtype=idx_t)
        i3 = np.zeros((nx - 1) * (ny - 1), dtype=idx_t)

        count = 0
        for yi in range(0, ny - 1):
            for xi in range(0, nx - 1):
                quad = np.array(
                    [
                        yi * nx + xi,
                        yi * nx + xi + 1,
                        (yi + 1) * nx + xi + 1,
                        (yi + 1) * nx + xi,
                    ],
                    dtype=idx_t,
                )

                i0[count] = quad[0]
                i1[count] = quad[1]
                i2[count] = quad[2]
                i3[count] = quad[3]

                count += 1

        idx.append(i0)
        idx.append(i1)
        idx.append(i2)
        idx.append(i3)

    elif cell_type in tri3_names:
        i0 = np.zeros(2 * (nx - 1) * (ny - 1), dtype=idx_t)
        i1 = np.zeros(2 * (nx - 1) * (ny - 1), dtype=idx_t)
        i2 = np.zeros(2 * (nx - 1) * (ny - 1), dtype=idx_t)

        count = 0
        for yi in range(0, ny - 1):
            for xi in range(0, nx - 1):
                tri = np.array(
                    [yi * nx + xi, yi * nx + xi + 1, (yi + 1) * nx + xi], dtype=idx_t
                )

                i0[count] = tri[0]
                i1[count] = tri[1]
                i2[count] = tri[2]
                count += 1

                tri = np.array(
                    [yi * nx + xi + 1, (yi + 1) * nx + xi + 1, (yi + 1) * nx + xi],
                    dtype=idx_t,
                )

                i0[count] = tri[0]
                i1[count] = tri[1]
                i2[count] = tri[2]
                count += 1

        idx.append(i0)
        idx.append(i1)
        idx.append(i2)
    else:
        print(f"Invalid cell_type {cell_type}")
        sys.exit(1)

    return idx, points


if __name__ == "__main__":
    argv = sys.argv

    usage = f"usage: {argv[0]} <output_foler>"

    if len(argv) < 2:
        print(usage)
        sys.exit(1)

    output_folder = argv[1]
    cell_type = "quad"
    nx = 2
    ny = 3
    w = 1
    h = 1

    try:
        opts, args = getopt.getopt(
            argv[2:], "c:x:y:", ["cell_type=", "width=", "height="]
        )
    except getopt.GetoptError as err:
        print(err)
        print(usage)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(usage)
            sys.exit()
        elif opt in ("-c", "--cell_type"):
            cell_type = arg
        elif opt in ("-x"):
            nx = int(arg)
        elif opt in ("-y"):
            ny = int(arg)
        elif opt in ("--height"):
            h = int(arg)
        elif opt in ("--width"):
            w = int(arg)
        else:
            print(f"Unused option {opt} = {arg}")
            sys.exit(1)

    if not os.path.exists(output_folder):
        os.mkdir(f"{output_folder}")

    print(f"nx={nx} ny={ny} width={w} height={h}")

    idx, points = create(w, h, nx, ny, cell_type)

    prefix = ["x", "y", "z"]
    for d in range(0, len(points)):
        points[d].tofile(f"{output_folder}/{prefix[d]}.raw")

    for d in range(0, len(idx)):
        idx[d].tofile(f"{output_folder}/i{d}.raw")
