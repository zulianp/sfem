#!/usr/bin/env python3

import numpy as np
import meshio
import taichi as ti

from time import perf_counter

t1_start = perf_counter()

# ti.init(arch=ti.arm64)
ti.init(arch=ti.gpu)

real_t = np.float32
idx_t = np.int32

mesh = meshio.read("chair.obj")

x =  mesh.points[:,0]
y =  mesh.points[:,1]
z =  mesh.points[:,2]

xmin = np.min(x).astype(real_t)
xmax = np.max(x).astype(real_t)

ymin = np.min(y).astype(real_t)
ymax = np.max(y).astype(real_t)

zmin = np.min(z).astype(real_t)
zmax = np.max(z).astype(real_t)

hmax = 0.5

x_range = xmax - xmin
y_range = ymax - ymin
z_range = zmax - zmin

nx = np.int64(np.ceil((x_range)/hmax))
ny = np.int64(np.ceil((y_range)/hmax))
nz = np.int64(np.ceil((z_range)/hmax)) + 1

num_points = len(x)

print(f'grid 	{nx} x {ny} x {nz}')
print(f'points 	{num_points}')

gx = np.linspace(xmin, xmax, nx).astype(real_t) 
gy = np.linspace(ymin, ymax, ny).astype(real_t) 
gz = np.linspace(zmin, zmax, nz).astype(real_t) 

xdiff = np.subtract.outer(gx, x)
ydiff = np.subtract.outer(gy, y)
zdiff = np.subtract.outer(gz, z)

xdiff = xdiff * xdiff
ydiff = ydiff * ydiff
zdiff = zdiff * zdiff


fxdiff = ti.field(ti.f32, shape=xdiff.shape)
fydiff = ti.field(ti.f32, shape=ydiff.shape)
fzdiff = ti.field(ti.f32, shape=zdiff.shape)

fxdiff.from_numpy(xdiff.astype(real_t))
fydiff.from_numpy(ydiff.astype(real_t))
fzdiff.from_numpy(zdiff.astype(real_t))

infty = real_t(np.max([x_range, y_range, z_range]))

edt = np.zeros((nz, ny, nx)).astype(real_t) + infty
print(f'shape {edt.shape}')

edt = ti.field(ti.f32, shape=edt.shape)

# @ti.kernel
# def compute():
# 	for k, j, i in ti.ndrange(nz, ny, nx):
# 		temp = infty
# 		for p in range(0, num_points):
# 			temp = ti.min(temp, fxdiff[i, p] + fydiff[j, p] + fzdiff[k, p])
# 		edt[k, j, i] = temp

# compute()


tix = ti.field(ti.f32, shape=x.shape)
tiy = ti.field(ti.f32, shape=y.shape)
tiz = ti.field(ti.f32, shape=z.shape)

hx = (nx - 1)
hy = (ny - 1)
hz = (nz - 1)

for b in mesh.cells:
	ncells, nnodesxelem = b.data.shape

	idx = ti.Vector.field(nnodesxelem, ti.i32, shape=ncells)
	idx.from_numpy(b.data.astype(idx_t))

	@ti.kernel
	def compute():
		for k, j, i, e in ti.ndrange(nz, ny, nx, ncells):
			temp = infty

			i0 = idx[e, 0]
			i1 = idx[e, 1]
			i2 = idx[e, 2]

			# Plane u, v
			ux = tix[i1] - tix[i0]
			uy = tiy[i1] - tiy[i0]
			uz = tiz[i1] - tiz[i0]
			norm_u = ti.sqrt(ux * ux + uy * uy + uz * uz)
			ux /= norm_u
			uy /= norm_u
			uz /= norm_u

			vx = tix[i2] - tix[i0]
			vy = tiy[i2] - tiy[i0]
			vz = tiz[i2] - tiz[i0]
			norm_v = ti.sqrt(vx * vx + vy * vy + vz * vz)
			ux /= norm_v
			uy /= norm_v
			uz /= norm_v

			# Plane normal
			nlx = (uy * vz) - (vy * uz)
          	nly = (uz * vx) - (vz * ux)
          	nlz = (ux * vy) - (vx * uy)

          	norm_nl = ti.sqrt(nlx * nlx + nly * nly + nlz * nlz)
          	nlx /= norm_nl
          	nly /= norm_nl
          	nlz /= norm_nl

          	# Distance of plane from origin
          	d = nlx * tix[i0] + nly * tiy[i0] + nlz * tiz[i0]

          	gpx = i  * hx
          	gpy = j  * hy
          	gpz = k  * hz

          	# Vector from triangle to point
          	dx = gpx - tix[i0] 
          	dy = gpy - tiy[i0] 
          	dz = gpz - tiz[i0] 

          	pp_dist = dx * nlx + dy * nly + dz * nlz - d

          	# Point in plane
          	pp_x = gpx - nlx * pp_dist
          	pp_y = gpy - nly * pp_dist
          	pp_z = gpz - nlz * pp_dist

          	uv_x = pp_x - tix[i0] 
          	uv_y = pp_y - tiy[i0] 
          	uv_z = pp_z - tiz[i0] 

          	s = (uv_x * ux + uv_y * uy + uv_z * uz) / norm_u
          	t = (uv_x * vx + uv_y * vy + uv_z * vz) / norm_v

          	dist_in_plane = 0
          	if s <= 0 and t <= 0:
          		# Behind origin
          		dist_in_plane = ti.sqrt(dx * dx + dy *dy + dz * dz)
          	elif s > 1 and t > 0 and t <= 1:
          		# distance to p1
          		# dist_in_plane = 
          	elif t > 0 and s > 0 and s <= 1:
          		# distance to p2
          		# 
          	elif s < 0 and t > 0:
          		# distance to v segment
          	elif t < 0 and s > 0:
          		# distance to u segment
          	if s + t <= 1:
          		# Inside the plane
          		temp = pp_dist
          	elif s 



			ti.atomic_min(edt[k, j, i], temp)

compute()


ti.sync()

t1_stop = perf_counter()
 

print("TTS:", t1_stop-t1_start)

edt.to_numpy().astype(real_t).tofile('edt.float32.raw')


