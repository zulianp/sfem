#!/usr/bin/env python3

import numpy as np
import sys, getopt
import os
import glob

idx_t = np.int32
geom_t = np.float32

def rectangle_mesh(argv):
    usage = f'usage: {argv[0]} <output_foler>'

    if(len(argv) < 2):
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
            argv[2:], "c:x:y:",
            ["cell_type=", "width=", "height="])
    except getopt.GetoptError as err:
        print(err)
        print(usage)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage)
            sys.exit()
        elif opt in ('-c', '--cell_type'):
            cell_type = arg
        elif opt in ('-x'):
            nx = int(arg)
        elif opt in ('-y'):
            ny = int(arg)
        elif opt in ('--height'):
            h = int(arg)
        elif opt in ('--width'):
            w = int(arg)
        else:
        	print(f'Unused option {opt} = {arg}')
        	sys.exit(1)

    if not os.path.exists(output_folder):
    	os.mkdir(f'{output_folder}')

    print(f'nx={nx} ny={ny} width={w} height={h}')

    gx = np.linspace(0, w, num=nx, dtype=geom_t)
    gy = np.linspace(0, h, num=ny, dtype=geom_t)

    x, y = np.meshgrid(gx, gy)

    x = np.reshape(x, x.shape[0] * x.shape[1])
    y = np.reshape(y, y.shape[0] * y.shape[1])
	
    i0 = np.zeros((nx - 1) * (ny - 1), dtype=idx_t)
    i1 = np.zeros((nx - 1) * (ny - 1), dtype=idx_t)
    i2 = np.zeros((nx - 1) * (ny - 1), dtype=idx_t)
    i3 = np.zeros((nx - 1) * (ny - 1), dtype=idx_t)

    count = 0
    for yi in range(0, ny-1):
    	for xi in range(0, nx-1):
    		quad = np.array([
    		yi * nx + xi,
    		yi * nx + xi + 1,
    		(yi+1) * nx + xi + 1,
    		(yi+1) * nx + xi], dtype=idx_t)

    		i0[count] = quad[0]
    		i1[count] = quad[1]
    		i2[count] = quad[2]
    		i3[count] = quad[3]

    		count += 1
    		
    i0.tofile(f'{output_folder}/i0.raw')
    i1.tofile(f'{output_folder}/i1.raw')
    i2.tofile(f'{output_folder}/i2.raw')
    i3.tofile(f'{output_folder}/i3.raw')

    x.tofile(f'{output_folder}/x.raw')
    y.tofile(f'{output_folder}/y.raw')

if __name__ == '__main__':
    rectangle_mesh(sys.argv)