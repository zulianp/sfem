#!/usr/bin/env python3

import numpy as np
import scipy as sp
import sys
import os
import math


idx_t = np.int32
real_t = np.float64
output_folder='eigs'

def main(argv):
	if len(argv) != 3:
		print(f'usage: {argv[0]} <crs_folder> <K>')
		exit(1)

	if not os.path.exists(output_folder):
		os.mkdir(f'{output_folder}')
	
	folder = argv[1]
	K = np.int32(argv[2])
	rowptr = np.fromfile(f'{folder}/rowptr.raw', dtype=idx_t)
	colidx = np.fromfile(f'{folder}/colidx.raw', dtype=idx_t)
	data   = np.ones(colidx.shape, dtype=real_t)
	N = len(rowptr) - 1

	K = min(N-2, K)

	print(f'num_vectors {K}')

	A = sp.sparse.csr_matrix((data, colidx, rowptr), shape=(N, N)) 
	vals, vecs = sp.sparse.linalg.eigs(A, K)

	for k in range(0, K):
		postfix='{num:05d}.raw'.format(num=k)
		name_real = f'{output_folder}/real.{postfix}'
		name_imag = f'{output_folder}/imag.{postfix}'
		vecs[:, k].real.tofile(name_real)
		vecs[:, k].imag.tofile(name_imag)


if __name__ == '__main__':
	main(sys.argv)