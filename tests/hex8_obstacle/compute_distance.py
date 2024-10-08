#!/usr/bin/env python3

import numpy as np
import sys

from sfem.sfem_config import *

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

	obs = -0.03 * (2 - x)**2 * (x + 2)**2
	obs += 0.1 * np.cos(x*2*np.pi)
	obs += 0.06 * np.cos(x*4*np.pi)
	obs  = 2 + obs[sideset]

	y = y[sideset]
	f = obs - y
	f.astype(dtype=real_t).tofile(path_output)
	print(f'minmax {np.min(f)} {np.max(f)}')
