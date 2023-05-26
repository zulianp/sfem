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
	if len(argv) != 4:
		print(f'usage: {argv[0]} <type=directed|undirected> <N> <output_folder>')
		exit(1)


	which = argv[1]
	N = int(argv[2])
	output_folder = argv[3]
	
	if not os.path.exists(output_folder):
		os.mkdir(f'{output_folder}')

	nnzxrow = 1
	if which == 'undirected':
		nnzxrow = 2
	elif which == 'laplacian':
		nnzxrow = 3
	elif which == 'odd':
		nnzxrow = 2
	

	rowptr = np.arange(0, N+1, 1, dtype=idx_t) * nnzxrow
	data = np.ones(N*nnzxrow, dtype=real_t)
	colidx = np.zeros(N*nnzxrow, dtype=idx_t)

	if which == 'undirected':
		for i in range(0, N):
			ip1 = (i+1) % N
			im1 = (i - 1 + N ) % N
			colidx[i*2] = im1
			colidx[i*2+1] = ip1
	elif which == 'laplacian':
		for i in range(0, N):
			im1 = (i - 1 + N) % N
			ip1 = (i + 1) % N
			
			colidx[i*3] = im1
			colidx[i*3+1] = i
			colidx[i*3+2] = ip1

			data[i*3] = 1
			data[i*3+1] = -2
			data[i*3+2] = 1
	elif which == 'odd':
		for i in range(0, N):
			ip1 = (i+1) % N
			im1 = (i - 1 + N ) % N
			colidx[i*2] = im1
			colidx[i*2+1] = ip1
			data[i*2] = -1
			data[i*2+1] = 1
	else:
		for i in range(0, N):
			colidx[i] = (i+1) % N

	# print(rowptr)
	# print(colidx)
	# print(data)

	rowptr.tofile(f'{output_folder}/rowptr.raw')
	colidx.tofile(f'{output_folder}/colidx.raw')
	data.tofile(f'{output_folder}/values.raw')

if __name__ == '__main__':
	main(sys.argv)
