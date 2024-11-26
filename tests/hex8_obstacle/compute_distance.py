#!/usr/bin/env python3

import numpy as np
import sys

from sfem.sfem_config import *

# import sfem.sfem_config as sc
# print(f'sfem.sfem_config {sc.__file__}')

if __name__ == '__main__':
	usage = f'{sys.argv[0]} <mesh> <sideset> <output>'

	if len(sys.argv) < 4:
		print(usage)
		exit(1)

	path_mesh = sys.argv[1]
	path_sideset = sys.argv[2]
	path_output = sys.argv[3]

	x = np.fromfile(f'{path_mesh}/x.raw', dtype=geom_t);
	y = np.fromfile(f'{path_mesh}/y.raw', dtype=geom_t);
	z = np.fromfile(f'{path_mesh}/z.raw', dtype=geom_t) - 0.5;
	sideset = np.fromfile(f'{path_sideset}', dtype=idx_t)


	# obs = 0.03 * (2 - x)**2 * (x + 2)**2
	# obs += 0.01 * np.cos(x*2*np.pi)
	# obs += 0.03 * np.cos(x*4*np.pi)
	# obs += 0.03 * np.cos(z*4*np.pi)
	# obs += 0.02 * np.cos(x*8*np.pi)
	# obs += 0.02 * np.cos(z*16*np.pi)
	# obs += 0.04 * np.cos(x*32*np.pi)
	# obs  = 1.37 + obs[sideset]
	# y = y[sideset]
	# f = obs - y


	indentation = 1
	wall = 1.5
	radius = ((0.5 - np.sqrt(x*x + z*z)))
	f = -0.1*np.cos(np.pi*2*radius) - 0.1
	f += -0.05*np.cos(np.pi*8*radius)
	f += -0.01*np.cos(np.pi*16*radius)
	parabola = -indentation * f + wall

	sdf = (parabola - y).astype(real_t)
	f = sdf[sideset]

	f.astype(dtype=real_t).tofile(path_output)
	print(f'n: {len(f)} real_t: {real_t}')
	print(f'minmax {np.min(f)} {np.max(f)}')
