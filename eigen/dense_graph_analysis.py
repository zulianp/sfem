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
output_folder='eigs'

def main(argv):
	if len(argv) != 4:
		print(f'usage: {argv[0]} <crs_folder> <which=LR|SR> <K>')
		exit(1)

	if not os.path.exists(output_folder):
		os.mkdir(f'{output_folder}')
	
	folder = argv[1]
	which = argv[2]
	K = np.int32(argv[3])
	rowptr = np.fromfile(f'{folder}/rowptr.raw', dtype=idx_t)
	colidx = np.fromfile(f'{folder}/colidx.raw', dtype=idx_t)
	data   = np.fromfile(f'{folder}/values.raw', dtype=real_t)
	uno    = np.ones(data.shape, dtype=real_t)

	N = len(rowptr) - 1

	A = np.zeros((N, N), dtype=real_t)

	for i in range(0, N):
		begin = rowptr[i]
		end = rowptr[i+1]

		cols = colidx[begin:end]
		vals = data[begin:end]
		A[i, cols] = vals

	print(f'num_vectors {N}')

	vals, vecs = la.eig(A)

	plt.plot(vals.real)
	# plt.plot(vals.imag)
	plt.title(f'Eigenvalues')
	plt.xlabel('Number')
	plt.ylabel('Value')
	plt.savefig('eigvals.png')

	min_val = vecs[0, 0]
	max_val = vecs[0, 0]

	for k in range(0, N):
		postfix='{num:05d}.raw'.format(num=k)
		name_real = f'{output_folder}/real.{postfix}'
		name_imag = f'{output_folder}/imag.{postfix}'
		vr = vecs[:, k].real
		vr.tofile(name_real)
		vecs[:, k].imag.tofile(name_imag)

		min_val = min(min_val, np.min(vr))
		max_val = max(max_val, np.max(vr))

	print(f'min_val {min_val}')
	print(f'max_val {max_val}')


if __name__ == '__main__':
	main(sys.argv)
