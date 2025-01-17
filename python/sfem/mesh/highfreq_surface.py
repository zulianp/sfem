#!/usr/bin/env python3

import numpy as np
import sys, getopt
import os
import glob


try:
    import rectangle_mesh
except ImportError:
    import sfem.mesh.rectangle_mesh as rectangle_mesh

if __name__ == '__main__':
    argv = sys.argv
    usage = f'usage: {argv[0]} <output_foler>'

    if(len(argv) < 2):
        print(usage)
        sys.exit(1)

    output_folder = argv[1]
    cell_type = "tri3"
    nx = 2
    ny = 2

    w = 1
    h = 1
    t = 1
    tx = 0
    ty = 0
    tz = 0
    skip_boundary = False;

    try:
        opts, args = getopt.getopt(
            argv[2:], "c:x:y:z:",
            ["cell_type=", "width=", "height=", "depth=", "tx=", "ty=", "tz=", "skip_boundary="])
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
            h = float(arg)
        elif opt in ('--width'):
            w = float(arg)
        elif opt in ('--depth'):
            t = float(arg)
        elif opt in ('--tx'):
            tx = float(arg)
        elif opt in ('--ty'):
            ty = float(arg)
        elif opt in ('--tz'):
            tz = float(arg)
        elif opt in ('--skip_boundary'):
            skip_boundary = int(arg)
        else:
            print(f'Unused option {opt} = {arg}')
            sys.exit(1)

    if not os.path.exists(output_folder):
        os.mkdir(f'{output_folder}')

    print(f'nx={nx} ny={ny} width={w} height={h} depth={t}')

    idx, points = rectangle_mesh.create(w, h, nx, ny, cell_type)

    x = points[0]
    y = points[1]

    center = tx + w / 2
    indentation = 1
    wall = 0
    radius = ((center - np.sqrt(x*x + y*y)))
    f = -0.1*np.cos(np.pi*2*radius) - 0.1
    f += -0.05*np.cos(np.pi*8*radius)
    f += -0.01*np.cos(np.pi*16*radius)
    f += -0.005*np.cos(np.pi*32*radius)
    f += -0.01*x * np.cos(np.pi/2 + -np.pi*8*x)
    f += 0.01*y * np.cos(np.pi/2 + -np.pi*8*y)
    f += 0.04*y*x * np.cos(np.pi/2 + -np.pi*32*np.exp(-y*y*x*x/10))
    f += 0.04*radius * np.cos(np.pi/2 + -np.pi*32*np.exp(-radius/10))
    z = -indentation * f + wall

    points.append(z)

    points[0] += tx
    points[1] += ty
    points[2] += tz

    meta = {}
    meta['element_type'] = rectangle_mesh.to_sfem_element_type(cell_type)
    meta['spatial_dimension'] = 3
    meta['rpath'] = True

    meta['n_points'] = len(points[0])
    meta['points'] = []
    prefix = ['x', 'y', 'z']
    for d in range(0, len(points)):
        
        meta['points'].append({prefix[d] : f'{prefix[d]}.raw'})

        path = f'{output_folder}/{prefix[d]}.raw'
        points[d].tofile(path)

    meta['elem_num_nodes'] = len(idx)
    meta['n_elements'] = len(idx[0])
    meta['elements'] = []
    for d in range(0, len(idx)):
        meta['elements'].append({f'i{d}': f'i{d}.raw'})
        path = f'{output_folder}/i{d}.raw'
        idx[d].tofile(path)

