#!/usr/bin/env python3

import numpy as np
import sys, getopt
import os
import glob
from rectangle_mesh import rectangle_mesh

idx_t = np.int32
geom_t = np.float32

def box_mesh(argv):
	usage = f'usage: {argv[0]} <output_foler>'

	if(len(argv) < 2):
		print(usage)
		sys.exit(1)

	output_folder = argv[1]
	cell_type = "quad"
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
			h = int(arg)
		elif opt in ('--width'):
			w = int(arg)
		elif opt in ('--depth'):
			t = int(arg)
		else:
			print(f'Unused option {opt} = {arg}')
			sys.exit(1)

	if cell_type == "quad" or cell_type == "triangle":
		rectangle_mesh(argv)
		return

	if not os.path.exists(output_folder):
		os.mkdir(f'{output_folder}')

	print(f'nx={nx} ny={ny} nz={nz} width={w} height={h} depth={t}')

	gx = np.linspace(0, w, num=nx, dtype=geom_t)
	gy = np.linspace(0, h, num=ny, dtype=geom_t)
	gz = np.linspace(0, t, num=nz, dtype=geom_t)

	x, y, z = np.meshgrid(gx, gy, gz)

	# ld = [ 1, nx, nx * ny ]
	ld = [ nz,  nz*nx, 1]

	x = np.reshape(x, x.shape[0] * x.shape[1] * x.shape[2])
	y = np.reshape(y, y.shape[0] * y.shape[1] * y.shape[2])
	z = np.reshape(z, z.shape[0] * z.shape[1] * z.shape[2])

	if cell_type == "hexahedron" or cell_type == "hex" or cell_type == "hex8":
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

		i0.tofile(f'{output_folder}/i0.raw')
		i1.tofile(f'{output_folder}/i1.raw')
		i2.tofile(f'{output_folder}/i2.raw')
		i3.tofile(f'{output_folder}/i3.raw')

		i4.tofile(f'{output_folder}/i4.raw')
		i5.tofile(f'{output_folder}/i5.raw')
		i6.tofile(f'{output_folder}/i6.raw')
		i7.tofile(f'{output_folder}/i7.raw')

	elif cell_type == "tetra" or cell_type == "tetrahedron" or cell_type == "tetra4" or cell_type == "tet4":
		ne = 5 * (nx - 1) * (ny - 1) * (nz - 1)

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

		i0.tofile(f'{output_folder}/i0.raw')
		i1.tofile(f'{output_folder}/i1.raw')
		i2.tofile(f'{output_folder}/i2.raw')
		i3.tofile(f'{output_folder}/i3.raw')
	else:
		print(f'Invalid cell_type {cell_type}')
		sys.exit(1)

	x.tofile(f'{output_folder}/x.raw')
	y.tofile(f'{output_folder}/y.raw')
	z.tofile(f'{output_folder}/z.raw')

if __name__ == '__main__':
	box_mesh(sys.argv)
