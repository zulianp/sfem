#!/usr/bin/env python3

import numpy as np
import scipy as sp
import sys
import os
import math
import matplotlib.pyplot as plt

idx_t = np.int32
real_t = np.float64

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def read_crs(folder):
	rowptr = np.fromfile(f'{folder}/rowptr.raw', dtype=idx_t)
	colidx = np.fromfile(f'{folder}/colidx.raw', dtype=idx_t)
	data   = np.fromfile(f'{folder}/values.raw', dtype=real_t)
	N = len(rowptr) - 1
	A = sp.sparse.csr_matrix((data, colidx, rowptr), shape=(N, N)) 
	return A

def read_block_crs(rb, cb, folder, export_folder=None):
	block_rowptr = np.fromfile(f'{folder}/rowptr.raw', dtype=idx_t)
	block_colidx = np.fromfile(f'{folder}/colidx.raw', dtype=idx_t)

	block_nnz = len(block_colidx)
	N = len(block_rowptr) - 1

	nnz = rb * cb * block_nnz
	data = np.zeros(nnz, dtype=real_t)

	nrows = rb * N
	rowptr = np.zeros(rb * N + 1, dtype=idx_t)-1
	colidx = np.zeros(nnz, dtype=idx_t)-1

	rowptr[0:N] = block_rowptr[0:N] * cb
	for r in range(1, rb):
		prev = (r-1) * N
		s = r * N
		e = s + N
		rowptr[s:e]  = rowptr[prev:s]
		rowptr[s:e] += block_nnz  * cb

	last = block_rowptr[N] - block_rowptr[N - 1]
	rowptr[nrows] = rowptr[nrows - 1] + (last * cb)

	assert(rowptr[nrows]  == nnz)

	for r in range(0, rb):
		for i in range(0, N):
			br_begin  = block_rowptr[i]
			br_end    = block_rowptr[i + 1]
			br_extent = br_end - br_begin

			r_offset = r * N + i
			r_begin = rowptr[r_offset]
			r_end   = rowptr[r_offset + 1]
			r_extent = r_end - r_begin

			for c in range(0, cb):
				s = r_begin + c * br_extent
				e = s + br_extent
				colidx[s:e] = block_colidx[br_begin:br_end] + c * N

	for r in range(0, rb):
		for c in range(0, cb):
			path = f'{folder}/values.{r*cb+c}.raw'
			print(path)
			d = np.fromfile(path, dtype=real_t)
			s = (r * cb + c) * block_nnz
			e = s + block_nnz
			data[s:e] = d

	if export_folder != None:
		mkdir(export_folder)

		rowptr.tofile(f'{export_folder}/rowptr.raw')
		colidx.tofile(f'{export_folder}/colidx.raw')
		data.tofile(f'{export_folder}/values.raw')

	A = sp.sparse.csr_matrix((data, colidx, rowptr), shape=(N*rb, N*cb)) 
	return A

if __name__ == '__main__':
	argv = sys.argv
	if len(argv) != 5:
		print(f'usage {argv[0]} <br> <bc> <folder> <output>')
		exit(1)

	read_block_crs(int(argv[1]), int(argv[2]), argv[3], argv[4])


# def read_block_crs(rb, cb, folder):
# 	rowptr = np.fromfile(f'{folder}/rowptr.raw', dtype=idx_t)
# 	colidx = np.fromfile(f'{folder}/colidx.raw', dtype=idx_t)

# 	nnz = len(colidx)
# 	N = len(rowptr) - 1

# 	data = np.zeros((rb, cb, nnz))
# 	for r in range(0, rb):
# 		for c in range(0, cb):
# 			path = f'{folder}/values.{r*cb+c}.raw'
# 			print(path)
# 			d = np.fromfile(path, dtype=real_t)
# 			data[r, c, :] = d
	
# 	data_t = data.transpose()

# 	print(N)
# 	print(colidx.shape)
# 	print(data_t.shape)

# 	A = sp.sparse.bsr_matrix((data_t, colidx, rowptr), blocksize=(rb,cb), shape=(N, N)) 
# 	return A
