#!/usr/bin/env python3

import numpy as np
import meshio
import taichi as ti
import sys

from time import perf_counter

vec3 = ti.math.vec3

try:
    from sfem_config import *
except NameError:
    print("mesh_to_sdf: self contained mode")
    from distance_point_to_triangle import *
    import smesh

    geom_t = np.float32

sdf_t = geom_t


class Grid:
    def __init__(self, nx, ny, nz, xmin, ymin, zmin, xmax, ymax, zmax):

        self.nx = np.int64(nx)
        self.ny = np.int64(ny)
        self.nz = np.int64(nz)

        self.xrange = np.float32(xmax - xmin)
        self.yrange = np.float32(ymax - ymin)
        self.zrange = np.float32(zmax - zmin)

        self.hx = np.float32(self.xrange / (nx - 1))
        self.hy = np.float32(self.yrange / (ny - 1))
        self.hz = np.float32(self.zrange / (nz - 1))


def udf(mesh, hmax, margin):
    t1_start = perf_counter()

    # ti.init(arch=ti.arm64)
    ti.init(arch=ti.gpu)

    x = mesh.points[:, 0].astype(geom_t)
    y = mesh.points[:, 1].astype(geom_t)
    z = mesh.points[:, 2].astype(geom_t)

    xmin = np.min(x).astype(geom_t) - margin
    xmax = np.max(x).astype(geom_t) + margin

    ymin = np.min(y).astype(geom_t) - margin
    ymax = np.max(y).astype(geom_t) + margin

    zmin = np.min(z).astype(geom_t) - margin
    zmax = np.max(z).astype(geom_t) + margin

    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin

    nx = np.int64(np.ceil((x_range) / hmax)) + 1
    ny = np.int64(np.ceil((y_range) / hmax)) + 1
    nz = np.int64(np.ceil((z_range) / hmax)) + 1

    print(f"hmax={hmax} margin={margin}")
    print(f"hmax={x_range} y_range={y_range} z_range={z_range}")

    num_points = len(x)

    print(f"grid 	{nx} x {ny} x {nz}")
    print(f"grid 	[{xmin}, {xmax}] x [{ymin}, {ymax}] x [{zmin}, {zmax}] ")
    print(f"points 	{num_points}")

    infty = sdf_t(np.max([x_range, y_range, z_range]) * 1000)
    edt = np.zeros((nz, ny, nx)).astype(sdf_t)

    print(f"shape {edt.shape}")

    edt = ti.field(ti.f32, shape=edt.shape)
    # sign = ti.field(ti.f32, shape=edt.shape)

    tix = ti.field(ti.f32, shape=x.shape)
    tiy = ti.field(ti.f32, shape=y.shape)
    tiz = ti.field(ti.f32, shape=z.shape)

    tix.from_numpy(x)
    tiy.from_numpy(y)
    tiz.from_numpy(z)

    hx = x_range / (nx - 1)
    hy = y_range / (ny - 1)
    hz = z_range / (nz - 1)

    for b in mesh.cells:
        ncells, nnodesxelem = b.data.shape
        print(f"{ncells} x {nnodesxelem}")

        ii0 = b.data[:, 0]
        ii1 = b.data[:, 1]
        ii2 = b.data[:, 2]

        idx0 = ti.field(ti.i32, shape=ii0.shape)
        idx0.from_numpy(ii0.astype(idx_t))

        idx1 = ti.field(ti.i32, shape=ii1.shape)
        idx1.from_numpy(ii1.astype(idx_t))

        idx2 = ti.field(ti.i32, shape=ii2.shape)
        idx2.from_numpy(ii2.astype(idx_t))

        @ti.kernel
        def compute():
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
                    i2 = idx2[e]

                    gpx = xmin + i * hx
                    gpy = ymin + j * hy
                    gpz = zmin + k * hz

                    p0 = vec3(tix[i0], tiy[i0], tiz[i0])
                    p1 = vec3(tix[i1], tiy[i1], tiz[i1])
                    p2 = vec3(tix[i2], tiy[i2], tiz[i2])

                    p = [gpx, gpy, gpz]
                    t = [p0, p1, p2]
                    q, __, __, __ = point_to_triangle(p, t)
                    d = ti.math.distance(p, q)

                    if d < e_min:
                        e_min = d

                    # 	n = ti.math.cross(ti.math.normalize(p1 - p0),  ti.math.normalize(p2 - p0))
                    # 	if d == 0:
                    # 		e_sign = 1
                    # 	elif ti.math.dot(p - q, n) < 0:
                    # 		e_sign = -1
                    # 	else:
                    # 		e_sign = 1

                    # if(ti.math.isnan(d)):
                    # 	print(f'Error = {tix[i0]} {tiy[i0]} {tiz[i0]}')
                    # 	print('\n')
                    # 	# print(f"Error NaN p={p}, t={t}, q={q}, d={d}")
                    # 	continue

                # edt[k, j, i] = e_sign * e_min
                edt[k, j, i] = e_min

    compute()
    ti.sync()

    grid = Grid(nx, ny, nz, xmin, ymin, zmin, xmax, ymax, zmax)

    t1_stop = perf_counter()
    print("TTS:", t1_stop - t1_start)

    nedt = edt.to_numpy().astype(sdf_t)
    print(f"d in [{np.min(nedt[:])}, {np.max(nedt[:])}]")
    return grid, nedt


if __name__ == "__main__":

    input_path = sys.argv[1]
    hmax = float(sys.argv[2])
    margin = float(sys.argv[3])
    output_path = sys.argv[4]

    # hmax = 0.005
    # margin = 4*hmax
    # input_path = "teapot.obj"
    # output_path = 'edt.float32.raw'

    mesh = meshio.read(input_path)
    grid, nedt = udf(mesh, hmax, margin)
    nedt.tofile(output_path)
