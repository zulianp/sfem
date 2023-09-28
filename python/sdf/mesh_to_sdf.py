#!/usr/bin/env python3

import numpy as np
import meshio
import taichi as ti
import sdf2
import sys

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

	print(f'hmax={hmax} margin={margin}')
	print(f'hmax={x_range} y_range={y_range} z_range={z_range}')

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

	tinx = ti.field(ti.f32, shape=x.shape)
	tiny = ti.field(ti.f32, shape=y.shape)
	tinz = ti.field(ti.f32, shape=z.shape)
	# area = ti.field(ti.f32, shape=x.shape)

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
		def compute_vertex_normals():
			for e in range(0, ncells):
				i0 = idx0[e]
				i1 = idx1[e]
				i2 = idx2[e]

				p0 = vec3(tix[i0], tiy[i0], tiz[i0])
				p1 = vec3(tix[i1], tiy[i1], tiz[i1])
				p2 = vec3(tix[i2], tiy[i2], tiz[i2])

				n = ti.math.cross(p1 - p0,  p2 - p0)
				a = ti.math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])

				ti.atomic_add(tinx[i0], n[0])
				ti.atomic_add(tiny[i0], n[1])
				ti.atomic_add(tinz[i0], n[2])

				ti.atomic_add(tinx[i1], n[0])
				ti.atomic_add(tiny[i1], n[1])
				ti.atomic_add(tinz[i1], n[2])

				ti.atomic_add(tinx[i2], n[0])
				ti.atomic_add(tiny[i2], n[1])
				ti.atomic_add(tinz[i2], n[2])

				# area[i0] += a
				# area[i1] += a
				# area[i2] += a

		compute_vertex_normals()

	# @ti.kernel
	# def in_place_e_div(n, num, denum):
	# 	for e in range(0, n):
	# 		num[e] /= denum[e]

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


					p = [ gpx, gpy, gpz ]
					t = [ p0, p1, p2 ]
					q, phi1, phi2 = sdf2.point_to_triangle(p, t)
					d = ti.math.distance(p, q)

					if d < e_min:
						e_min = d

						phi0 = 1 - phi1 - phi2
						
						# Interior
						n = ti.math.cross(p1 - p0,  p2 - p0)
						a = ti.math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
						n /= a

						tol = 1e-16
						isvertex = approxeq(phi0, 1, tol) or approxeq(phi1, 1, tol) or approxeq(phi2, 1, tol)
						# isedge   = approxeq(phi0, 0, tol) or approxeq(phi1, 0, tol) or approxeq(phi2, 0, tol)

						if isvertex:
						# if isvertex or isedge:
							# Corners	
							n0 = vec3(tinx[i0], tinx[i0], tinx[i0])
							n1 = vec3(tinx[i1], tinx[i1], tinx[i1])
							n2 = vec3(tinx[i2], tinx[i2], tinx[i2])
							n = ti.math.normalize(phi0 * n0 + phi1 * n1 + phi2 * n2)
						# else:
						# 	# Edge

						if d == 0:
							e_sign = 1
						elif ti.math.dot(p - q, n) < 0:
							e_sign = -1
						else:
							e_sign = 1

					if(ti.math.isnan(d)):
						print(f'Error = {tix[i0]} {tiy[i0]} {tiz[i0]}')
					# 	print('\n')
					# 	# print(f"Error NaN p={p}, t={t}, q={q}, d={d}")
					# 	continue

				edt[k, j, i] = e_sign * e_min
				# edt[k, j, i] = e_min

		compute_sdf()
	ti.sync()

	t1_stop = perf_counter()
	print("TTS:", t1_stop - t1_start)

	tinx.to_numpy().astype(real_t).tofile('nx.float32.raw')
	tiny.to_numpy().astype(real_t).tofile('ny.float32.raw')
	tinz.to_numpy().astype(real_t).tofile('nz.float32.raw')

	nedt = edt.to_numpy().astype(real_t)
	print(f'd in [{np.min(nedt[:])}, {np.max(nedt[:])}]')
	return nedt

if __name__ == '__main__':

	input_path = sys.argv[1]
	hmax = float(sys.argv[2])
	margin = float(sys.argv[3])
	output_path = sys.argv[4]

	# hmax = 0.005
	# margin = 4*hmax
	# input_path = "teapot.obj"
	# output_path = 'edt.float32.raw'

	mesh = meshio.read(input_path)
	nedt = sdt(mesh, hmax, margin)
	nedt.tofile(output_path)
