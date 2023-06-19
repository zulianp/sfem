import numpy as np
import scipy as sp
import sys
import os
import math
import matplotlib.pyplot as plt

idx_t = np.int32
real_t = np.float64

def read_crs(folder)
	rowptr = np.fromfile(f'{folder}/rowptr.raw', dtype=idx_t)
	colidx = np.fromfile(f'{folder}/colidx.raw', dtype=idx_t)
	data   = np.fromfile(f'{folder}/values.raw', dtype=real_t)
	N = len(rowptr) - 1
	A = sp.sparse.csr_matrix((data, colidx, rowptr), shape=(N, N)) 
	return A

def read_block_crs(rb, cb, folder):
	rowptr = np.fromfile(f'{folder}/rowptr.raw', dtype=idx_t)
	colidx = np.fromfile(f'{folder}/colidx.raw', dtype=idx_t)

	nnz = len(colidx)
	N = len(rowptr) - 1

	data = np.zeros((rb, rc, nnz))
	for r in range(0, rb):
		for c in range(0, cb):
			path = f'{folder}/values.{r*cb+c}.raw'
			d = np.fromfile(path, dtype=real_t)
			data[r, c, :] = d
			
	A = sp.sparse.bsr_matrix((data, colidx, rowptr), shape=(N, N)) 
	return A
