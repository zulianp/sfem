#!/usr/bin/env python3

import numpy as np
import meshio
import taichi as ti
import sdf2

from time import perf_counter
vec3 = ti.math.vec3

def sdt(mesh, hmax, margin):
	t1_start = perf_counter()

	# ti.init(arch=ti.arm64)
	ti.init(arch=ti.gpu)

	real_t = np.float32
	idx_t = np.int32

	x =  mesh.points[:,0].astype(real_t)
	y =  mesh.points[:,1].astype(real_t)
	z =  mesh.points[:,2].astype(real_t)

	xmin = np.min(x).astype(real_t) - margin
	xmax = np.max(x).astype(real_t) + margin

	ymin = np.min(y).astype(real_t) - margin
	ymax = np.max(y).astype(real_t) + margin

	zmin = np.min(z).astype(real_t) - margin
	zmax = np.max(z).astype(real_t) + margin



	x_range = xmax - xmin
	y_range = ymax - ymin
	z_range = zmax - zmin

	nx = np.int64(np.ceil((x_range)/hmax)) + 1
	ny = np.int64(np.ceil((y_range)/hmax)) + 1
	nz = np.int64(np.ceil((z_range)/hmax)) + 1

	num_points = len(x)

	print(f'grid 	{nx} x {ny} x {nz}')
	print(f'grid 	[{xmin}, {xmax}] x [{ymin}, {ymax}] x [{zmin}, {zmax}] ')
	print(f'points 	{num_points}')

	infty = real_t(np.max([x_range, y_range, z_range]) * 1000)
	edt = np.zeros((nz, ny, nx)).astype(real_t)

	print(f'shape {edt.shape}')

	edt = ti.field(ti.f32, shape=edt.shape)
	# sign = ti.field(ti.f32, shape=edt.shape)

	tix = ti.field(ti.f32, shape=x.shape)
	tiy = ti.field(ti.f32, shape=y.shape)
	tiz = ti.field(ti.f32, shape=z.shape)

	tix.from_numpy(x)
	tiy.from_numpy(y)
	tiz.from_numpy(z)

	hx = x_range/(nx - 1)
	hy = y_range/(ny - 1)
	hz = z_range/(nz - 1)

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

		@ti.kernel
		def compute():
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


					p = [ gpx, gpy, gpz ]
					t = [ p0, p1, p2 ]
					q = sdf2.point_to_triangle(p, t)
					d = ti.math.distance(p, q)

					if d < e_min:
						e_min = d
						# n = ti.math.cross(ti.math.normalize(p1 - p0),  ti.math.normalize(p2 - p0))
						# if d == 0:
						# 	e_sign = 0
						# elif ti.math.dot(q - p0, n) > 0:
						# 	e_sign = -1
						# else:
						# 	e_sign = 1

					if(ti.math.isnan(d)):
						print(f'Error = {tix[i0]} {tiy[i0]} {tiz[i0]}')
						print('\n')
						# print(f"Error NaN p={p}, t={t}, q={q}, d={d}")
						continue

				# edt[k, j, i] = e_sign * e_min
				edt[k, j, i] = e_min

	compute()
	ti.sync()

	t1_stop = perf_counter()
	print("TTS:", t1_stop - t1_start)

	nedt = edt.to_numpy().astype(real_t)
	print(f'd in [{np.min(nedt[:])}, {np.max(nedt[:])}]')
	return nedt

hmax = 0.005
input_path = "teapot.obj"
output_path = 'edt.float32.raw'
mesh = meshio.read(input_path)
nedt = sdt(mesh, hmax, 4*hmax)
nedt.tofile(output_path)
