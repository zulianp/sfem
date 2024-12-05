#!/usr/bin/env python3

import numpy as np
import sys, getopt
import os
import glob

tet4_names = ("tetra", "tetrahedron", "tetra4", "tet4", "TET4")
hex8_names = ("hexahedron", "hex", "hex8", "HEX8")

try:
    import rectangle_mesh
except ImportError:
    import sfem.mesh.rectangle_mesh as rectangle_mesh

idx_t = np.int32
geom_t = np.float32

def leading_dim(nx, ny, nz):
    return [nz,  nz*nx, 1]

def create_face_idx(n, nxf):
    return np.zeros((n,  nxf), dtype=idx_t)

def create_boundary_faces(cell_type, nx, ny, nz):
    ld = leading_dim(nx, ny, nz)


    fact = 2
    nxf = 3

    left = create_face_idx(fact*(ny-1)*(nz-1),  nxf)
    right = create_face_idx(fact*(ny-1)*(nz-1), nxf)

    top = create_face_idx(fact*(nx-1)*(nz-1), nxf)
    bottom = create_face_idx(fact*(nx-1)*(nz-1), nxf)

    front = create_face_idx(nx*ny, nxf)
    back = create_face_idx(nx*ny, nxf)

    idx = 0
    for zi in range(0, nz-1):
        for yi in range(0, ny-1):

            xl = 0 * ld[0]
            xr = (nx - 1) * ld[0]

            y0 = yi * ld[1]
            z0 = zi * ld[2]
            p0 = y0 + z0

            y1 = (yi + 1) * ld[1]
            z1 = zi * ld[2]
            p1 = y1 + z1

            y2 = (yi + 1) * ld[1]
            z2 = (zi + 1) * ld[2]
            p2 = y2 + z2

            y3 = yi * ld[1]
            z3 = (zi+1) * ld[2]
            p3 = y3 + z3

            quad_left  = [ xl + p0, xl + p1, xl + p2, xl + p3 ]
            quad_right = [ xr + p0, xr + p1, xr + p2, xr + p3 ]

            left[idx, 0]  = quad_left[0]
            left[idx, 1]  = quad_left[2]
            left[idx, 2]  = quad_left[1]


            right[idx, 0]  = quad_right[0]
            right[idx, 1]  = quad_right[1]
            right[idx, 2]  = quad_right[2]

            idx += 1

            left[idx, 0]  = quad_left[0]
            left[idx, 1]  = quad_left[3]
            left[idx, 2]  = quad_left[2]

            right[idx, 0]  = quad_right[0]
            right[idx, 1]  = quad_right[2]
            right[idx, 2]  = quad_right[3]

            idx += 1
        
            

    # idx = 0 
    # for zi in range(0, nz):
    #     for xi in range(0, nx):
            
    #         yb = 0 * ld[1]
    #         yt = (ny - 1) * ld[1]

    #         xp = yi * ld[1]
    #         zp = zi * ld[2]

    #         p = xp + zp

    #         bottom[idx]  = yb + p
    #         top[idx]     = yt + p

    #         idx += 1

    # idx = 0 
    # for yi in range(0, ny):
    #     for xi in range(0, nx):
            
    #         zf = 0 * ld[2]
    #         zb = (nz - 1) * ld[2]

    #         xp = yi * ld[1]
    #         yp = yi * ld[1]

    #         p = xp + yp

    #         front[idx] = zf + p
    #         back[idx]  = zb + p

    #         idx += 1

    # print(left)
    return {
        "left"      : left,
        "right"     : right
        # ,
        # "top"       : top,
        # "bottom"    : bottom,
        # "front"     : front,
        # "back"      : back
    }

def create_boundary_nodes(nx, ny, nz):
    ld = leading_dim(nx, ny, nz)

    left = np.zeros(ny*nz, dtype=idx_t)
    right = np.zeros(ny*nz, dtype=idx_t)

    top = np.zeros(nx*nz, dtype=idx_t)
    bottom = np.zeros(nx*nz, dtype=idx_t)

    front = np.zeros(nx*ny, dtype=idx_t)
    back = np.zeros(nx*ny, dtype=idx_t)

    idx = 0
    for zi in range(0, nz):
        for yi in range(0, ny):

            xl = 0 * ld[0]
            xr = (nx - 1) * ld[0]

            yp = yi * ld[1]
            zp = zi * ld[2]
            p = yp + zp

            left[idx]  = xl + p
            right[idx] = xr + p

            idx += 1

    idx = 0 
    for zi in range(0, nz):
        for xi in range(0, nx):
            
            yb = 0 * ld[1]
            yt = (ny - 1) * ld[1]

            xp = yi * ld[1]
            zp = zi * ld[2]

            p = xp + zp

            bottom[idx]  = yb + p
            top[idx]     = yt + p

            idx += 1

    idx = 0 
    for yi in range(0, ny):
        for xi in range(0, nx):
            
            zf = 0 * ld[2]
            zb = (nz - 1) * ld[2]

            xp = yi * ld[1]
            yp = yi * ld[1]

            p = xp + yp

            front[idx] = zf + p
            back[idx]  = zb + p

            idx += 1

    return {
        "left"      : left,
        "right"     : right,
        "top"       : top,
        "bottom"    : bottom,
        "front"     : front,
        "back"      : back
    }


def create(w, h, t, nx, ny, nz, cell_type):
    cell_type = cell_type.lower()
    
    gx = np.linspace(0, w, num=nx, dtype=geom_t)
    gy = np.linspace(0, h, num=ny, dtype=geom_t)
    gz = np.linspace(0, t, num=nz, dtype=geom_t)

    x, y, z = np.meshgrid(gx, gy, gz)

    # ld = [ 1, nx, nx * ny ]
    # ld = [ nz,  nz*nx, 1]
    ld = leading_dim(nx, ny, nz)

    x = np.reshape(x, x.shape[0] * x.shape[1] * x.shape[2])
    y = np.reshape(y, y.shape[0] * y.shape[1] * y.shape[2])
    z = np.reshape(z, z.shape[0] * z.shape[1] * z.shape[2])

    idx = []
    points = [x, y, z]

    if cell_type in hex8_names:
        ne = (nx - 1) * (ny - 1) * (nz - 1)

        i0 = np.zeros(ne, dtype=idx_t)
        i1 = np.zeros(ne, dtype=idx_t)
        i2 = np.zeros(ne, dtype=idx_t)
        i3 = np.zeros(ne, dtype=idx_t)

        i4 = np.zeros(ne, dtype=idx_t)
        i5 = np.zeros(ne, dtype=idx_t)
        i6 = np.zeros(ne, dtype=idx_t)
        i7 = np.zeros(ne, dtype=idx_t)

        count = 0
        for zi in range(0, nz-1):
            for yi in range(0, ny-1):
                for xi in range(0, nx-1):
                    x0 = xi * ld[0]
                    y0 = yi * ld[1]
                    z0 = zi * ld[2]

                    x1 = (xi + 1) * ld[0]
                    y1 = (yi + 1) * ld[1]
                    z1 = (zi + 1) * ld[2]

                    hexa = np.array([
                        # Bottom
                        x0 + y0 + z0, # 1 (0, 0, 0)
                        x1 + y0 + z0, # 2 (1, 0, 0)
                        x1 + y1 + z0, # 3 (1, 1, 0)
                        x0 + y1 + z0, # 4 (0, 1, 0)
                        # Top
                        x0 + y0 + z1, # 5 (0, 0, 1)
                        x1 + y0 + z1, # 6 (1, 0, 1)
                        x1 + y1 + z1, # 7 (1, 1, 1)
                        x0 + y1 + z1  # 8 (0, 1, 1)
                        ], dtype=idx_t)

                    i0[count] = hexa[0]
                    i1[count] = hexa[1]
                    i2[count] = hexa[2]
                    i3[count] = hexa[3]

                    i4[count] = hexa[4]
                    i5[count] = hexa[5]
                    i6[count] = hexa[6]
                    i7[count] = hexa[7]

                    count += 1

        idx.append(i0)
        idx.append(i1)
        idx.append(i2)
        idx.append(i3)

        idx.append(i4)
        idx.append(i5)
        idx.append(i6)
        idx.append(i7)

    elif cell_type in tet4_names:
        use_incompatibile_tet = False
        if use_incompatibile_tet:
            ne = 5 * (nx - 1) * (ny - 1) * (nz - 1)
        else:
            ne = 6 * (nx - 1) * (ny - 1) * (nz - 1)

        i0 = np.zeros(ne, dtype=idx_t)
        i1 = np.zeros(ne, dtype=idx_t)
        i2 = np.zeros(ne, dtype=idx_t)
        i3 = np.zeros(ne, dtype=idx_t)

        count = 0
        for zi in range(0, nz-1):
            for yi in range(0, ny-1):
                for xi in range(0, nx-1):
                    x0 = xi * ld[0]
                    y0 = yi * ld[1]
                    z0 = zi * ld[2]

                    x1 = (xi + 1) * ld[0]
                    y1 = (yi + 1) * ld[1]
                    z1 = (zi + 1) * ld[2]

                    hexa = np.array([
                        # Bottom
                        x0 + y0 + z0, # 1 (0, 0, 0)
                        x1 + y0 + z0, # 2 (1, 0, 0)
                        x1 + y1 + z0, # 3 (1, 1, 0)
                        x0 + y1 + z0, # 4 (0, 1, 0)
                        # Top
                        x0 + y0 + z1, # 5 (0, 0, 1)
                        x1 + y0 + z1, # 6 (1, 0, 1)
                        x1 + y1 + z1, # 7 (1, 1, 1)
                        x0 + y1 + z1  # 8 (0, 1, 1)
                        ], dtype=idx_t)

                    if use_incompatibile_tet:

                        # Tet 0
                        i0[count] = hexa[0]
                        i1[count] = hexa[1]
                        i2[count] = hexa[2]
                        i3[count] = hexa[5]
                        count += 1

                        # Tet 1
                        i0[count] = hexa[0]
                        i1[count] = hexa[5]
                        i2[count] = hexa[7]
                        i3[count] = hexa[4]
                        count += 1

                        # Tet 2
                        i0[count] = hexa[2]
                        i1[count] = hexa[5]
                        i2[count] = hexa[6]
                        i3[count] = hexa[7]
                        count += 1

                        # Tet 3
                        i0[count] = hexa[0]
                        i1[count] = hexa[2]
                        i2[count] = hexa[7]
                        i3[count] = hexa[5]
                        count += 1

                        # Tet 4
                        i0[count] = hexa[0]
                        i1[count] = hexa[2]
                        i2[count] = hexa[3]
                        i3[count] = hexa[7]
                        count += 1

                    else:
                        # First prism

                        # Tet 0
                        i0[count] = hexa[0]
                        i1[count] = hexa[1]
                        i2[count] = hexa[3]
                        i3[count] = hexa[7]
                        count += 1

                        # Tet 1
                        i0[count] = hexa[0]
                        i1[count] = hexa[1]
                        i2[count] = hexa[7]
                        i3[count] = hexa[5]
                        count += 1

                        # Tet 2
                        i0[count] = hexa[0]
                        i1[count] = hexa[4]
                        i2[count] = hexa[5]
                        i3[count] = hexa[7]
                        count += 1

                        # second prism

                        # Tet 3
                        i0[count] = hexa[1]
                        i1[count] = hexa[2]
                        i2[count] = hexa[3]
                        i3[count] = hexa[6]
                        count += 1

                        # Tet 4
                        i0[count] = hexa[1]
                        i1[count] = hexa[3]
                        i2[count] = hexa[7]
                        i3[count] = hexa[6]
                        count += 1

                        # Tet 5
                        i0[count] = hexa[1]
                        i1[count] = hexa[5]
                        i2[count] = hexa[6]
                        i3[count] = hexa[7]
                        count += 1

        idx.append(i0)
        idx.append(i1)
        idx.append(i2)
        idx.append(i3)
    else:
        print(f'Invalid cell_type {cell_type}')
        sys.exit(1)

    return idx, points

if __name__ == '__main__':
    argv = sys.argv
    usage = f'usage: {argv[0]} <output_foler>'

    if(len(argv) < 2):
        print(usage)
        sys.exit(1)

    output_folder = argv[1]
    cell_type = "tet4"
    nx = 2
    ny = 2
    nz = 2
    w = 1
    h = 1
    t = 1

    try:
        opts, args = getopt.getopt(
            argv[2:], "c:x:y:z:",
            ["cell_type=", "width=", "height=", "depth="])
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
        elif opt in ('-z'):
            nz = int(arg)
        elif opt in ('--height'):
            h = float(arg)
        elif opt in ('--width'):
            w = float(arg)
        elif opt in ('--depth'):
            t = float(arg)
        else:
            print(f'Unused option {opt} = {arg}')
            sys.exit(1)

    if not os.path.exists(output_folder):
        os.mkdir(f'{output_folder}')

    print(f'nx={nx} ny={ny} nz={nz} width={w} height={h} depth={t}')

    if cell_type == "quad" or cell_type == "triangle":
        idx, points = rectangle_mesh.create(w, h, nx, ny, cell_type)
    else:
        idx, points = create(w, h, t, nx, ny, nz, cell_type)

    prefix = ['x', 'y', 'z']
    for d in range(0, len(points)):
        points[d].tofile(f'{output_folder}/{prefix[d]}.raw')

    for d in range(0, len(idx)):
        idx[d].tofile(f'{output_folder}/i{d}.raw')

    boundary_nodes_dir = f'{output_folder}/boundary_nodes'
    if not os.path.exists(boundary_nodes_dir):
        os.mkdir(f'{boundary_nodes_dir}')

    boundary_nodes = create_boundary_nodes(nx, ny, nz)
    for k, v in boundary_nodes.items():
        name = f'{k}.{str(idx_t.__name__)}.raw'
        v.tofile(f'{boundary_nodes_dir}/{name}')


    boundary_faces_dir = f'{output_folder}/surface'
    if not os.path.exists(boundary_faces_dir):
        os.mkdir(f'{boundary_faces_dir}')

    boundary_faces = create_boundary_faces(cell_type, nx, ny, nz)
    for k, v in boundary_faces.items():
        path = f'{boundary_faces_dir}/{k}'
        if not os.path.exists(path):
            os.mkdir(f'{path}')


        n, nxf = v.shape
        print(f'{k}: {path} {n} {nxf}')
        for d in range(0, nxf):
            idx_path = f'{path}/i{d}.{str(idx_t.__name__)}.raw'
            v[:, d].flatten().tofile(idx_path)
