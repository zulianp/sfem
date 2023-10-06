#!/usr/bin/env python3

import numpy as np
import meshio
import taichi as ti
import sdf2
import sys
import smesh

from time import perf_counter
vec3 = ti.math.vec3

@ti.func
def cross(u, v):
	return -ti.math.cross(u, v)

def edge_normals(mesh, dual):
	# Compute edge normals
	row_ptr = dual.row_ptr
	idx = dual.idx

	enx = ti.field(ti.f32, shape=idx.shape)
	eny = ti.field(ti.f32, shape=idx.shape)
	enz = ti.field(ti.f32, shape=idx.shape)

	for b in mesh.cells:
		ncells, nnodesxelem = b.data.shape

		@ti.kernel
		def edge_normals_kernel():
			for e in ti.ndrange(ncells):
				print(e)



def sdt(mesh, dual, hmax, margin):
	t1_start = perf_counter()

	# ti.init(arch=ti.arm64)
	ti.init(arch=ti.gpu)

	edge_normals(mesh, dual)

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

	dual_row_ptr = ti.field(ti.i32, shape=dual.row_ptr.shape)
	dual_idx = ti.field(ti.i32, shape=dual.idx.shape)

	dual_row_ptr.from_numpy(dual.row_ptr)
	dual_idx.from_numpy(dual.idx)
	
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

						# if sdf2.isvertex(entity):
						n0 = vec3(tinx[i0], tiny[i0], tinz[i0])
						n1 = vec3(tinx[i1], tiny[i1], tinz[i1])
						n2 = vec3(tinx[i2], tiny[i2], tinz[i2])
						phi0 = 1 - phi1 - phi2
						n = ti.math.normalize(phi0 * n0 + phi1 * n1 + phi2 * n2)
						# else:
						# 	n = cross(p1 - p0,  p2 - p0)
						# 	n = ti.math.normalize(n)

						# 	if sdf2.isedge(entity):
						# 		f0 = ti.math.ivec3(i0, i1, i2)
						# 		f1 = ti.math.ivec3(i0, i1, i2)
								
						# 		found = False
						# 		for rptr in range(dual_row_ptr[e], dual_row_ptr[e+1]):
						# 			e_adj = dual_idx[rptr]
						# 			f1 = ti.math.ivec3(idx0[e_adj], idx1[e_adj], idx2[e_adj])
						# 			if sdf2.are_adj(f0, f1):
						# 				found = True
						# 				break

						# 		if not found:
						# 			print("Error")

						# 		p_adj_0 = vec3(tix[f1[0]], tiy[f1[0]], tiz[f1[0]])
						# 		p_adj_1 = vec3(tix[f1[1]], tiy[f1[1]], tiz[f1[1]])
						# 		p_adj_2 = vec3(tix[f1[2]], tiy[f1[2]], tiz[f1[2]])

						# 		n_adj = cross(p_adj_1 - p_adj_0,  p_adj_2 - p_adj_0)
						# 		n_adj = ti.math.normalize(n_adj)
						# 		n += n_adj
						# 		n = ti.math.normalize(n)
						# 		print(f'Edge!')

						if d == 0:
							e_sign = 1
						elif ti.math.dot(p - q, n) < 0:
							e_sign = -1
						else:
							e_sign = 1

					# if(ti.math.isnan(d)):
					# 	print(f'Error = {tix[i0]} {tiy[i0]} {tiz[i0]}')

				edt[k, j, i] = e_sign * e_min
				# edt[k, j, i] = e_min

		compute_sdf()
	ti.sync()

	t1_stop = perf_counter()
	print("TTS:", t1_stop - t1_start)

	tinx.to_numpy().astype(real_t).tofile('nx.float32.raw')
	tiny.to_numpy().astype(real_t).tofile('ny.float32.raw')
	tinz.to_numpy().astype(real_t).tofile('nz.float32.raw')

	# grid_nx.to_numpy().astype(real_t).tofile('gnx.float32.raw')
	# grid_ny.to_numpy().astype(real_t).tofile('gny.float32.raw')
	# grid_nz.to_numpy().astype(real_t).tofile('gnz.float32.raw')

	nedt = edt.to_numpy().astype(real_t)
	print(f'd in [{np.min(nedt[:])}, {np.max(nedt[:])}]')
	return nedt

if __name__ == '__main__':
	if len(sys.argv) < 6:
		print(f'usage {sys.argv[0]} <mesh> <dual_graph> <hmax> <margin> <output_path>')
	else:
		input_path = sys.argv[1]
		input_dual = sys.argv[2]
		hmax = float(sys.argv[3])
		margin = float(sys.argv[4])
		output_path = sys.argv[5]

		mesh = smesh.read(input_path)
		dual = smesh.read_dual_graph(input_dual)

		# mesh = meshio.read(input_path)
		nedt = sdt(mesh, dual, hmax, margin)
		nedt.tofile(output_path)
