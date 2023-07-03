#!/usr/bin/env python3

import numpy as np
import sys

def main(argv):
	if len(argv) != 7:
		print(f'usage: {argv[0]} <x> <y> <z> <x.raw> <y.raw> <z.raw>')
		exit()

	x0 = float(argv[1])
	y0 = float(argv[2])
	z0 = float(argv[3])

	x = np.fromfile(argv[3+1], dtype=np.float32)
	y = np.fromfile(argv[3+2], dtype=np.float32)
	z = np.fromfile(argv[3+3], dtype=np.float32)

	n = len(x)

	dx = x - x0
	dy = y - y0
	dz = z - z0

	d = dx * dx + dy * dy + dz * dz

	i = np.argmin(d)

	print(i)

if __name__ == '__main__':
    main(sys.argv)

