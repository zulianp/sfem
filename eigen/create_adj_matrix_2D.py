#!/usr/bin/env python3

import numpy as np
from numpy import linalg as la

import scipy as sp
import sys
import os
import math
import matplotlib.pyplot as plt

idx_t = np.int32
real_t = np.float64

def main(argv):
	if len(argv) != 5:
		print(f'usage: {argv[0]} <type=directed|undirected> <nx> <ny> <output_folder>')
		exit(1)

	which = argv[1]
	nx = int(argv[2])
	ny = int(argv[3])
	output_folder = argv[4]
	
	if not os.path.exists(output_folder):
		os.mkdir(f'{output_folder}')

	nnzxrow = 1
	if which == 'undirected':
		nnzxrow = 4
	elif which == 'laplacian':
		nnzxrow = 5
	elif which == 'odd':
		nnzxrow = 4
	elif which == 'yonly':
		nnzxrow = 1
	
	N = nx * ny 
	rowptr = np.arange(0, N+1, 1, dtype=idx_t) * nnzxrow
	data = np.ones(N*nnzxrow, dtype=real_t)
	colidx = np.zeros(N*nnzxrow, dtype=idx_t)

	if which == 'undirected':
		for x in range(0, nx):
			xp1 = (x+1) % nx
			xm1 = (x - 1 + nx ) % nx

			for y in range(0, ny):
				yp1 = (y + 1) % ny
				ym1 = (y - 1 + ny ) % ny
				i = x * ny + y
				offset = i * 4
				colidx[offset] = xm1 * ny + y
				colidx[offset+1] = xp1 * ny + y
				colidx[offset+2] = x * ny + ym1
				colidx[offset+3] = x * ny + yp1
	elif which == 'laplacian':
		for x in range(0, nx):
			xp1 = (x+1) % nx
			xm1 = (x - 1 + nx ) % nx

			for y in range(0, ny):
				yp1 = (y + 1) % ny
				ym1 = (y - 1 + ny ) % ny
				i = x * ny + y
				offset = i * 5
				colidx[offset] = x * ny + y
				colidx[offset+1] = xm1 * ny + y
				colidx[offset+2] = xp1 * ny + y
				colidx[offset+3] = x * ny + ym1
				colidx[offset+4] = x * ny + yp1

				data[offset]   = -4
				data[offset+1] = 1
				data[offset+2] = 1
				data[offset+3] = 1
				data[offset+4] = 1
	elif which == 'yonly':
		for y in range(0, ny):
			yp1 = (y + 1) % ny
			ym1 = (y - 1 + ny ) % ny

			for x in range(0, nx):
				xp1 = (x + 1) % nx
				xm1 = (x - 1 + nx) % nx
				# i = x * ny + y
				i = x + y * nx

				if nnzxrow == 1:
					# colidx[i] = x * ny + yp1
					colidx[i] = yp1 * nx + x
	else:
		for x in range(0, nx):
			xp1 = (x + 1) % nx
			xm1 = (x - 1 + nx) % nx

			for y in range(0, ny):
				yp1 = (y + 1) % ny
				ym1 = (y - 1 + ny ) % ny
				i = x * ny + y

				if nnzxrow == 1:
					colidx[i] = xp1 * ny + yp1
				elif nnzxrow == 3:
					offset = i * 3
					colidx[offset] = xp1 * ny + y
					colidx[offset+1] = x * ny + yp1
					colidx[offset+2] = xp1 * ny + yp1
				else:
					assert nnzxrow == 2
					offset = i * 2
					colidx[offset] = xp1 * ny + y
					colidx[offset+1] = x * ny + yp1

	rowptr.tofile(f'{output_folder}/rowptr.raw')
	colidx.tofile(f'{output_folder}/colidx.raw')
	data.tofile(f'{output_folder}/values.raw')

if __name__ == '__main__':
	main(sys.argv)
