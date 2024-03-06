#!/usr/bin/env python3

import numpy as np
import sys

def main(argv):
	if len(argv) != 6:
		print(f'usage: {argv[0]} <expr> <x.raw> <y.raw> <z.raw> <out.raw>')
		exit()

	expr = argv[1]

	x = np.fromfile(argv[2], dtype=np.float32).astype(np.float64)
	y = np.fromfile(argv[3], dtype=np.float32).astype(np.float64)
	z = np.fromfile(argv[4], dtype=np.float32).astype(np.float64)
	f = eval(expr)
	f.astype(np.float64).tofile(argv[5])

if __name__ == '__main__':
    main(sys.argv)

