#!/usr/bin/env python3

import meshio
import numpy as np
import os

def write_case(path, i0, i1, i2, i3, x, y, z):
	if not os.path.exists(path):
		os.makedirs(path)

	i0.tofile(f'{path}/i0.raw')
	i1.tofile(f'{path}/i1.raw')
	i2.tofile(f'{path}/i2.raw')
	i3.tofile(f'{path}/i3.raw')

	x.tofile(f'{path}/x.raw')
	y.tofile(f'{path}/y.raw')
	z.tofile(f'{path}/z.raw')

	points = np.array([x, y, z]).transpose()
	cells = [
	    ("tetra", np.array([i0, i1, i2, i3]).transpose())
	]

	mesh = meshio.Mesh(points, cells)
	mesh.write(f'{path}/mesh.e')

# Fixed
i0 = np.array([0,4],dtype=np.int32)
i1 = np.array([1,1],dtype=np.int32)
i2 = np.array([2,2],dtype=np.int32)
i3 = np.array([3,3],dtype=np.int32)

#############################################

x = np.array([0, 1, 0, 0, 1],dtype=np.float32)
y = np.array([0, 0, 1, 0, 1],dtype=np.float32)
z = np.array([0, 0, 0, 1, 1],dtype=np.float32)

write_case('meshes/2', i0, i1, i2, i3, x, y, z)


# Fixed
i0 = np.array([0,4,0],dtype=np.int32)
i1 = np.array([1,1,1],dtype=np.int32)
i2 = np.array([2,2,2],dtype=np.int32)
i3 = np.array([3,3,5],dtype=np.int32)

#############################################

x = np.array([0, 1, 0, 0, 1, 0],dtype=np.float32)
y = np.array([0, 0, 1, 0, 1, 0],dtype=np.float32)
z = np.array([0, 0, 0, 1, 1, -1],dtype=np.float32)

write_case('meshes/3', i0, i1, i2, i3, x, y, z)

