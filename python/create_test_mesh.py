#!/usr/bin/env python3

import meshio
import numpy as np
import sys

def main(argv):
	output = "./"

	if(len(argv) > 1):
		output = argv[1]

	if False:
	# if True:
		mesh = meshio.read('./mesh.e')
	else:
		xfactor = 1.0
		yfactor = 1.0
		zfactor = 1.0

		if(len(argv) > 2):
			xfactor = float(argv[2])

		if(len(argv) > 3):
			yfactor = float(argv[3])

		if(len(argv) > 4):
			zfactor = float(argv[4])

		# two triangles and one quad
		points = [
		    [0.0, 0.0, 0],
		    [xfactor, 0.0, 0],
		    [0.0, yfactor, 0],
		    [0.0, 0.0, zfactor]
		]

		cells = [
		    ("tetra", [[0, 1, 2, 3]]),
		]

		mesh = meshio.Mesh(
		    points,
		    cells,
	)

	n_cells=0
	for b in mesh.cells:
		d = b.data
		n_cells += len(d)

	i0 = np.zeros(n_cells, dtype=np.int32)
	i1 = np.zeros(n_cells, dtype=np.int32)
	i2 = np.zeros(n_cells, dtype=np.int32)
	i3 = np.zeros(n_cells, dtype=np.int32)

	idx=0
	for b in mesh.cells:
		for elem in b.data:
			i0[idx] = elem[0]
			i1[idx] = elem[1]
			i2[idx] = elem[2]
			i3[idx] = elem[3]
			assert( elem[0] >= 0)
			idx += 1

	###################################
	# Export indices
	###################################

	# First node of each element
	i0.astype(np.int32).tofile(f'{output}/i0.raw')

	# Second node of each element
	i1.astype(np.int32).tofile(f'{output}/i1.raw')

	# ...
	i2.astype(np.int32).tofile(f'{output}/i2.raw')

	# ...
	i3.astype(np.int32).tofile(f'{output}/i3.raw')

	###################################
	# Points
	###################################

	xyz = np.transpose(mesh.points)
	x = xyz[0, :].astype(np.float32)
	y = xyz[1, :].astype(np.float32)
	z = xyz[2, :].astype(np.float32)

	x.tofile(f'{output}/x.raw')
	y.tofile(f'{output}/y.raw')
	z.tofile(f'{output}/z.raw')

	print(n_cells)

if __name__ == '__main__':
    main(sys.argv)
