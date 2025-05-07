#!/usr/bin/env python3

import sys
import os

# import smesh
from time import perf_counter

import numpy as np
import meshio
import taichi as ti

ti.init(arch=ti.gpu)
# ti.init(arch=ti.cpu)
vec3 = ti.math.vec3

try:
    from sfem_config import *
except ModuleNotFoundError:
    print("mesh_to_edf: self contained mode")
    from distance_point_to_triangle import *
    import smesh

    geom_t = np.float32
    idx_t = np.int32

edf_t = geom_t


@ti.func
def cross(u, v):
    return -ti.math.cross(u, v)


def read_mesh(input_path):
    fname, fextension = os.path.splitext(input_path)

    print(f"file extension {fextension}")
    if fextension in [".e", ".exo", ".vtk"]:
        mesh = meshio.read(input_path)
    else:
        # Use sfem mesh format here
        mesh = smesh.read(input_path)
    return mesh

def compute_aabb(mesh, margin=0):
    x = mesh.points[:, 0].astype(geom_t)
    y = mesh.points[:, 1].astype(geom_t)
    z = mesh.points[:, 2].astype(geom_t)

    pmin = [0, 0, 0]
    pmax = [0, 0, 0]

    pmin[0] = np.min(x).astype(geom_t) - margin
    pmax[0] = np.max(x).astype(geom_t) + margin

    pmin[1] = np.min(y).astype(geom_t) - margin
    pmax[1] = np.max(y).astype(geom_t) + margin

    pmin[2] = np.min(z).astype(geom_t) - margin
    pmax[2] = np.max(z).astype(geom_t) + margin
    return np.array(pmin), np.array(pmax)


def mesh_to_edf(mesh, pmin, pmax, hmax):
    t1_start = perf_counter()

    x = mesh.points[:, 0].astype(geom_t)
    y = mesh.points[:, 1].astype(geom_t)
    z = mesh.points[:, 2].astype(geom_t)

    xmin = pmin[0]
    xmax = pmax[0]

    ymin = pmin[1]
    ymax = pmax[1]

    zmin = pmin[2]
    zmax = pmax[2]

    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin

    nx = np.int64(np.ceil((x_range) / hmax)) + 1
    ny = np.int64(np.ceil((y_range) / hmax)) + 1
    nz = np.int64(np.ceil((z_range) / hmax)) + 1

    print(f"hmax={hmax} margin={margin}")
    print(f"hmax={x_range} y_range={y_range} z_range={z_range}")

    num_points = len(x)

    print(f"grid    {nx} x {ny} x {nz}")
    print(f"grid    [{xmin}, {xmax}] x [{ymin}, {ymax}] x [{zmin}, {zmax}] ")
    print(f"points  {num_points}")

    infty = edf_t(np.max([x_range, y_range, z_range]) * 1000)
    edt = np.zeros((nz, ny, nx)).astype(edf_t)

    print(f"shape {edt.shape}")

    edt = ti.field(ti.f32, shape=edt.shape)
    tix = ti.field(ti.f32, shape=x.shape)
    tiy = ti.field(ti.f32, shape=y.shape)
    tiz = ti.field(ti.f32, shape=z.shape)

    tix.from_numpy(x)
    tiy.from_numpy(y)
    tiz.from_numpy(z)

    hx = x_range / (nx - 1)
    hy = y_range / (ny - 1)
    hz = z_range / (nz - 1)

    tinx = ti.field(ti.f32, shape=x.shape)
    tiny = ti.field(ti.f32, shape=y.shape)
    tinz = ti.field(ti.f32, shape=z.shape)

    for b in mesh.cells:
        ncells, nnodesxelem = b.data.shape
        print(f"{ncells} x {nnodesxelem}")

        ii0 = b.data[:, 0]
        ii1 = b.data[:, 1]

        idx0 = ti.field(ti.i32, shape=ii0.shape)
        idx0.from_numpy(ii0.astype(idx_t))

        idx1 = ti.field(ti.i32, shape=ii1.shape)
        idx1.from_numpy(ii1.astype(idx_t))

        @ti.func
        def approxeq(a, b, tol):
            v = a - b
            v *= ti.math.sign(v)
            return v < tol

        @ti.func
        def point_to_segment_distance(p, a, b):
            """Returns the minimum distance between point p and segment ab."""
            ab = b - a
            ap = p - a
            
            t = ti.math.dot(ap, ab) / ti.math.dot(ab, ab)
            
            if t > 1:
                t = 1

            if t < 0:
                t = 0

            closest = a + t * ab
            d = p - closest
            return ti.math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])

        @ti.kernel
        def compute_edf():
            for (
                k,
                j,
                i,
            ) in ti.ndrange(nz, ny, nx):
                e_min = infty
                e_sign = 1

                for e in range(0, ncells):
                    temp = infty

                    i0 = idx0[e]
                    i1 = idx1[e]

                    gpx = xmin + i * hx
                    gpy = ymin + j * hy
                    gpz = zmin + k * hz

                    p0 = vec3(tix[i0], tiy[i0], tiz[i0])
                    p1 = vec3(tix[i1], tiy[i1], tiz[i1])


                    p = [gpx, gpy, gpz]
                    d = point_to_segment_distance(p, p0, p1)

                    if d < e_min:
                        e_min = d

                edt[k, j, i] = e_min

        compute_edf()
    ti.sync()

    t1_stop = perf_counter()
    print("TTS:", t1_stop - t1_start)

    nedt = edt.to_numpy().astype(edf_t)
    print(f"d in [{np.min(nedt[:])}, {np.max(nedt[:])}]")
    return nedt, [nx, ny, nz]


if __name__ == "__main__":
    import sys, getopt

    usage = f"{sys.argv[0]}.py <mesh> <out>"

    infty = 1e8
    pmin = [infty, infty, infty]
    pmax = [-infty, -infty, -infty]

    hmax = 1
    margin = 0
    scale_box = 1

    if len(sys.argv) < 3:
        print(usage)
        exit(1)
    try:
        opts, args = getopt.getopt(
            sys.argv[3 : len(sys.argv)],
            "h",
            [
                "help",
                "xmin=",
                "ymin=",
                "zmin=",
                "xmax=",
                "ymax=",
                "zmax=",
                "hmax=",
                "margin=",
                "box_from_mesh=",
                "scale_box=",
                "export_normals",
            ],
        )

    except getopt.GetoptError as err:
        print(err)
        print(usage)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
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
        elif opt in ("--scale_box"):
            scale_box = float(arg)
        

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    mesh = read_mesh(input_path)
    pmin, pmax = compute_aabb(mesh, margin)

    if scale_box != 1:
        pmean = (pmin + pmax) / 2
        ppmin = pmin - pmean
        ppmax = pmax - pmean
        ppmin *= scale_box
        ppmax *= scale_box
        pmin = ppmin + pmean
        pmax = ppmax + pmean

    nedt, dims = mesh_to_edf(mesh, pmin, pmax, hmax)
    nedt.tofile(output_path)

    header = f"spatial_dimension: 3\n"
    header += f"nx: {dims[0]}\n"
    header += f"ny: {dims[1]}\n"
    header += f"nz: {dims[2]}\n"
    header += f"block_size: 1\n"
    header += f"type: float\n"
    header += f"ox: {pmin[0]}\n"
    header += f"oy: {pmin[1]}\n"
    header += f"oz: {pmin[2]}\n"
    header += f"dx: {(pmax[0] - pmin[0])/(dims[0] - 1)}\n"
    header += f"dy: {(pmax[1] - pmin[1])/(dims[1] - 1)}\n"
    header += f"dz: {(pmax[2] - pmin[2])/(dims[2] - 1)}\n"
    header += f'rpath: 0\n'
    header += f"path: {os.path.abspath(output_path)}\n"

    fname, fextension = os.path.splitext(output_path)
    pdir = os.path.dirname(fname)

    if pdir == "":
        pdir = "./"

    fname = os.path.basename(fname)

    with open(f"{pdir}/metadata_{fname}.yml", "w") as f:
        f.write(header)
