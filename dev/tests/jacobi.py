#!/usr/bin/env python3

import numpy as np
import scipy as sp

rowptr = np.fromfile("rowptr.raw", dtype=np.int32)
colidx = np.fromfile("colidx.raw", dtype=np.int32)
values = np.fromfile("values.raw", dtype=np.float64)
rhs = np.fromfile("rhs.raw", dtype=np.float64)
diag = np.fromfile("diag.raw", dtype=np.float64)

rows = len(rowptr)-1

A = sp.sparse.csr_matrix((values, colidx, rowptr), shape=(rows, rows)) 

x = np.zeros(rows)

inv_diag = 1./diag

# print(diag)

r = rhs - A * x
x += r

def jacobi(r, x):
	xnew = np.zeros(x.shape)
	for i in range(0, rows):
		b = rowptr[i]
		e = rowptr[i+1]

		acc = r[i]
		for k in range(b, e):
			j = colidx[k]
			v = values[k]
			if j != i:
				acc -= v * x[j]
		xnew[i] = acc / diag[i]
	return xnew

def gauss_seidel(r, x):
	for i in range(0, rows):
		b = rowptr[i]
		e = rowptr[i+1]

		acc = r[i]
		for k in range(b, e):
			j = colidx[k]
			v = values[k]
			# if j != i: # Use sum later to remove diagonal entry check
			acc -= v * x[j]
		x[i] += acc / diag[i]
	return x

r = rhs - A * x
norm_r = np.sqrt(np.sum(r * r))
print(f'{0}) {norm_r}')

for i in range(0, 100):
	# r = rhs - A * x
	# x += 0.001 * r
	# x += 0.95 * inv_diag * r
	x = gauss_seidel(rhs, x)
	# x = jacobi(rhs, x)


	r = rhs - A * x
	norm_r = np.sqrt(np.sum(r * r))
	print(f'{i+1}) {norm_r}')
	