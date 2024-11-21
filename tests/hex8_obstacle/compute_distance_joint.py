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

	obs =  0.3

	coord = x[sideset]
	f = obs - coord
	f.astype(dtype=real_t).tofile(path_output)
	print(f'n: {len(f)} real_t: {real_t}')
	print(f'minmax {np.min(f)} {np.max(f)}')
