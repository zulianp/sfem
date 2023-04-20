#!/usr/bin/env python3
import numpy as np
import sys
import os

geom_t = os.getenv('py_sfem_geom_t')
real_t = os.getenv('py_sfem_real_t')

if not geom_t:
	geom_t = np.float32

if not real_t:
	real_t = np.float64

def main(argv):
	if(len(argv) < 6):
		print(f'usage {argv[0]} <x.raw> <y.raw> <z.raw> <expr> <out.raw>')

	x = np.fromfile(sys.argv[1], dtype=geom_t)
	y = np.fromfile(sys.argv[2], dtype=geom_t)
	z = np.fromfile(sys.argv[3], dtype=geom_t)
	fx = eval(sys.argv[4])
	fx.astype(real_t).tofile(sys.argv[5])

if "__main__" == __name__:
	main(sys.argv)
