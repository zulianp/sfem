#!/usr/bin/env python3

import numpy as np
import meshio
import taichi as ti
import sdf2
import sys
import os
import smesh

from time import perf_counter

ti.init(arch=ti.gpu)
vec3 = ti.math.vec3
real_t = np.float32
idx_t = np.int32

@ti.func
def cross(u, v):
    return -ti.math.cross(u, v)

def read_mesh(input_path):
    fname, fextension = os.path.splitext(input_path)

    print(f'file extension {fextension}')
    if fextension in ['.e', '.exo', '.vtk']:
        mesh = meshio.read(input_path)
    else:
        # Use sfem mesh format here
        mesh = smesh.read(input_path)
    return mesh

def select_submesh(mesh, pmin, pmax):
    x =  mesh.points[:,0].astype(real_t)
    y =  mesh.points[:,1].astype(real_t)
    z =  mesh.points[:,2].astype(real_t)

    is_node_inside_min = np.logical_and(np.logical_and(x > pmin[0], y > pmin[1]), z > pmin[2])
    is_node_inside_max = np.logical_and(np.logical_and(x < pmax[0], y < pmax[1]), z < pmax[2])
    is_node_inside = np.logical_and(is_node_inside_min, is_node_inside_max)

    submesh = smesh.Mesh()
    submesh.points = mesh.points
    selected_cells = None
    for b in mesh.cells:
        ncells, nnodesxelem = b.data.shape
        print(f'{ncells} x {nnodesxelem}')

        i0 = b.data[:, 0].astype(np.int32)
        i1 = b.data[:, 1].astype(np.int32)
        i2 = b.data[:, 2].astype(np.int32)

        keep_cells = np.logical_or(np.logical_or(is_node_inside[i0], is_node_inside[i1]), is_node_inside[i2])

        selected_cells = b.data[keep_cells, :]
        submesh.add_cells(selected_cells)
        print(f'cells {selected_cells.shape[0]}/{ncells}')
    return submesh

def compute_aabb(mesh, margin=0):
    x =  mesh.points[:,0].astype(real_t)
    y =  mesh.points[:,1].astype(real_t)
    z =  mesh.points[:,2].astype(real_t)

    pmin = [0, 0, 0]
    pmax = [0, 0, 0]

    pmin[0] = np.min(x).astype(real_t) - margin
    pmax[0] = np.max(x).astype(real_t) + margin

    pmin[1] = np.min(y).astype(real_t) - margin
    pmax[1] = np.max(y).astype(real_t) + margin

    pmin[2] = np.min(z).astype(real_t) - margin
    pmax[2] = np.max(z).astype(real_t) + margin
    return np.array(pmin), np.array(pmax)

def sdt(mesh, pmin, pmax, hmax):
    t1_start = perf_counter()

    x =  mesh.points[:,0].astype(real_t)
    y =  mesh.points[:,1].astype(real_t)
    z =  mesh.points[:,2].astype(real_t)

    xmin = pmin[0]
    xmax = pmax[0]

    ymin = pmin[1]
    ymax = pmax[1]

    zmin = pmin[2]
    zmax = pmax[2]

    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin

    nx = np.int64(np.ceil((x_range)/hmax)) + 1
    ny = np.int64(np.ceil((y_range)/hmax)) + 1
    nz = np.int64(np.ceil((z_range)/hmax)) + 1

    print(f'hmax={hmax} margin={margin}')
    print(f'hmax={x_range} y_range={y_range} z_range={z_range}')

    num_points = len(x)

    print(f'grid    {nx} x {ny} x {nz}')
    print(f'grid    [{xmin}, {xmax}] x [{ymin}, {ymax}] x [{zmin}, {zmax}] ')
    print(f'points  {num_points}')

    infty = real_t(np.max([x_range, y_range, z_range]) * 1000)
    edt = np.zeros((nz, ny, nx)).astype(real_t)

    print(f'shape {edt.shape}')

    edt = ti.field(ti.f32, shape=edt.shape)
    tix = ti.field(ti.f32, shape=x.shape)
    tiy = ti.field(ti.f32, shape=y.shape)
    tiz = ti.field(ti.f32, shape=z.shape)

    tix.from_numpy(x)
    tiy.from_numpy(y)
    tiz.from_numpy(z)

    hx = x_range/(nx - 1)
    hy = y_range/(ny - 1)
    hz = z_range/(nz - 1)

    tinx = ti.field(ti.f32, shape=x.shape)
    tiny = ti.field(ti.f32, shape=y.shape)
    tinz = ti.field(ti.f32, shape=z.shape)
    
    for b in mesh.cells:
        ncells, nnodesxelem = b.data.shape
        print(f'{ncells} x {nnodesxelem}')

        ii0 = b.data[:, 0]
        ii1 = b.data[:, 1]
        ii2 = b.data[:, 2]

        idx0 = ti.field(ti.i32, shape=ii0.shape)
        idx0.from_numpy(ii0.astype(idx_t))

        idx1 = ti.field(ti.i32, shape=ii1.shape)
        idx1.from_numpy(ii1.astype(idx_t))

        idx2 = ti.field(ti.i32, shape=ii2.shape)
        idx2.from_numpy(ii2.astype(idx_t))

        @ti.func
        def angle_triangle(p1, p2, p3):
            x1 = p1[0]
            x2 = p2[0]
            x3 = p3[0]

            y1 = p1[1]
            y2 = p2[1]
            y3 = p3[1]

            z1 = p1[2]
            z2 = p2[2]
            z3 = p3[2]

            num = (x2-x1)*(x3-x1)+(y2-y1)*(y3-y1)+(z2-z1)*(z3-z1)
            den = ti.math.sqrt(ti.math.pow((x2-x1),2)+ti.math.pow((y2-y1),2)+ti.math.pow((z2-z1),2))* ti.math.sqrt(ti.math.pow((x3-x1),2)+ti.math.pow((y3-y1),2)+pow((z3-z1),2))
            angle = ti.math.acos(num / den)
            return angle ;
            
        @ti.kernel
        def compute_vertex_normals():
            for e in range(0, ncells):
                i0 = idx0[e]
                i1 = idx1[e]
                i2 = idx2[e]

                p0 = vec3(tix[i0], tiy[i0], tiz[i0])
                p1 = vec3(tix[i1], tiy[i1], tiz[i1])
                p2 = vec3(tix[i2], tiy[i2], tiz[i2])

                n = cross(p1 - p0,  p2 - p0)
                n = ti.math.normalize(n)

                ti.atomic_add(tinx[i0], n[0])
                ti.atomic_add(tiny[i0], n[1])
                ti.atomic_add(tinz[i0], n[2])

                ti.atomic_add(tinx[i1], n[0])
                ti.atomic_add(tiny[i1], n[1])
                ti.atomic_add(tinz[i1], n[2])

                ti.atomic_add(tinx[i2], n[0])
                ti.atomic_add(tiny[i2], n[1])
                ti.atomic_add(tinz[i2], n[2])

        compute_vertex_normals()

    @ti.kernel
    def normalize():
        for e in range(0, num_points):
            v = ti.math.normalize(vec3(tinx[e], tiny[e], tinz[e]))

            tinx[e] = v[0]
            tiny[e] = v[1]
            tinz[e] = v[2]

    normalize()

    for b in mesh.cells:
        ncells, nnodesxelem = b.data.shape
        print(f'{ncells} x {nnodesxelem}')

        ii0 = b.data[:, 0]
        ii1 = b.data[:, 1]
        ii2 = b.data[:, 2]

        idx0 = ti.field(ti.i32, shape=ii0.shape)
        idx0.from_numpy(ii0.astype(idx_t))

        idx1 = ti.field(ti.i32, shape=ii1.shape)
        idx1.from_numpy(ii1.astype(idx_t))

        idx2 = ti.field(ti.i32, shape=ii2.shape)
        idx2.from_numpy(ii2.astype(idx_t))

        @ti.func
        def approxeq(a, b, tol):
            v = a - b
            v *= ti.math.sign(v)
            return v < tol

        @ti.kernel
        def compute_sdf():
            for k, j, i, in ti.ndrange(nz, ny, nx):
                e_min = infty
                e_sign = 1

                for e in range(0, ncells):
                    temp = infty

                    i0 = idx0[e]
                    i1 = idx1[e]
                    i2 = idx2[e]

                    gpx = xmin + i  * hx
                    gpy = ymin + j  * hy
                    gpz = zmin + k  * hz

                    p0 = vec3(tix[i0], tiy[i0], tiz[i0])
                    p1 = vec3(tix[i1], tiy[i1], tiz[i1])
                    p2 = vec3(tix[i2], tiy[i2], tiz[i2])
                    n  = vec3(0, 0, 0)

                    p = [ gpx, gpy, gpz ]
                    t = [ p0, p1, p2 ]
                    q, phi1, phi2, entity = sdf2.point_to_triangle(p, t)
                    d = ti.math.distance(p, q)

                    if d < e_min:
                        e_min = d
                        n0 = vec3(tinx[i0], tiny[i0], tinz[i0])
                        n1 = vec3(tinx[i1], tiny[i1], tinz[i1])
                        n2 = vec3(tinx[i2], tiny[i2], tinz[i2])
                        phi0 = 1 - phi1 - phi2
                        n = ti.math.normalize(phi0 * n0 + phi1 * n1 + phi2 * n2)

                        if d == 0:
                            e_sign = 1
                        elif ti.math.dot(p - q, n) < 0:
                            e_sign = -1
                        else:
                            e_sign = 1

                edt[k, j, i] = e_sign * e_min

        compute_sdf()
    ti.sync()

    t1_stop = perf_counter()
    print("TTS:", t1_stop - t1_start)

    # tinx.to_numpy().astype(real_t).tofile('nx.float32.raw')
    # tiny.to_numpy().astype(real_t).tofile('ny.float32.raw')
    # tinz.to_numpy().astype(real_t).tofile('nz.float32.raw')

    nedt = edt.to_numpy().astype(real_t)
    print(f'd in [{np.min(nedt[:])}, {np.max(nedt[:])}]')
    return nedt, [nx, ny, nz]

if __name__ == '__main__':
    import sys, getopt

    usage = f'{sys.argv[0]}.py <mesh> <out>'

    infty = 1e8
    pmin = [ infty,  infty,  infty]
    pmax = [-infty, -infty, -infty]

    hmax = 1
    margin = 0
    scale_box=1

    if(len(sys.argv) < 3):
        print(usage)
    try:
        opts, args = getopt.getopt(
            sys.argv[3:len(sys.argv)], "h",["help",  "xmin=", "ymin=", "zmin=", "xmax=", "ymax=", "zmax=", "hmax=", "margin=", "box_from_mesh=", "scale_box="])

    except getopt.GetoptError as err:
        print(err)
        print(usage)
        sys.exit(2)

    defined_bounds = False

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage)
            sys.exit()
        elif opt in ("--xmin"):
            pmin[0] = float(arg) 
            defined_bounds = True
        elif opt in ("-ymin"):
            pmin[1] = float(arg) 
            defined_bounds = True
        elif opt in ("--zmin"):
            pmin[2] = float(arg) 
            defined_bounds = True
        elif opt in ("--xmax"):
            pmax[0] = float(arg) 
            defined_bounds = True
        elif opt in ("-ymax"):
            pmax[1] = float(arg) 
            defined_bounds = True
        elif opt in ("--zmax"):
            pmax[2] = float(arg)  
        elif opt in ("--hmax"):
            hmax = float(arg)  
        elif opt in ("--margin"):
            margin = float(arg)  
        elif opt in ("--box_from_mesh"):
            box_from_mesh = arg
        elif opt in ("--scale_box"):
            scale_box = float(arg)

    if box_from_mesh != None:
        aux_mesh = read_mesh(arg)
        pmin, pmax = compute_aabb(aux_mesh, margin)
        defined_bounds = True
        aux_mesh = None

    if defined_bounds:
        found_error = False
        diagonstic = ""
        coord = ['x', 'y', 'z']
        for d in range(0, 3):
            if pmin[d] == -infty:
                found_error = True
                diagonstic += f"Missing coordinate {coord[d]} for minimum\n"
            if pmax[d] == -infty:
                found_error = True
                diagonstic += f"Missing coordinate {coord[d]} for maximum\n"
        if found_error:
            print(diagonstic)
            sys.exit(1)
   
    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    mesh = read_mesh(input_path)

    if not defined_bounds:
        pmin, pmax = compute_aabb(mesh, margin)

    if scale_box != 1:
        pmean = (pmin + pmax) / 2
        ppmin = pmin - pmean
        ppmax = pmax - pmean
        ppmin *= scale_box
        ppmax *= scale_box
        pmin = ppmin + pmean
        pmax = ppmax + pmean

    submesh = select_submesh(mesh, pmin, pmax)
    submesh.write('submesh.vtk')
    nedt, dims = sdt(submesh, pmin, pmax, hmax)
    nedt.tofile(output_path)

    header =    f'nx: {dims[0]}\n'
    header +=   f'ny: {dims[1]}\n'
    header +=   f'nz: {dims[2]}\n'
    header +=   f'block_size: 1\n'
    header +=   f'type: float\n'

    fname, fextension = os.path.splitext(output_path)
    pdir = os.path.dirname(fname)
    fname = os.path.basename(fname)
    
    with open(f'{pdir}/metadata_{fname}.yml', 'w') as f:
        f.write(header)

