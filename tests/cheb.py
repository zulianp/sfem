#!/usr/bin/env python3

import numpy as np
import scipy as sp

rowptr = np.fromfile("rowptr.raw", dtype=np.int32)
colidx = np.fromfile("colidx.raw", dtype=np.int32)
values = np.fromfile("values.raw", dtype=np.float64)
rhs = np.fromfile("rhs.raw", dtype=np.float64)
diag = np.fromfile("diag.raw", dtype=np.float64)

rows = len(rowptr)-1

mat = sp.sparse.csr_matrix((values, colidx, rowptr), shape=(rows, rows)) 
inv_diag = 1./diag

def pA(x):
	global mat
	global inv_diag

	# Left precond
	# t = mat * x
	# return inv_diag * t

	# Right precond
	t = inv_diag * x
	return mat * t

# A = sp.sparse.linalg.LinearOperator((rows,rows), matvec=pA)
A = mat


x = np.zeros(rows)



# Gradient descent step for BCs
r = rhs - A * x
x += r

def powermethod(A, rhs):
	ev = rhs
	ev = ev / np.sqrt(np.sum(ev*ev))

	lmbda = 0
	lmbda_old = 0

	for k in range(0, 10000):
		ev = A * ev
		lmbda = np.sqrt(np.sum(ev*ev))
		ev = ev / lmbda

		if np.abs(lmbda_old -lmbda) < 1e-5:
			print(f'{k}) lmbda = {lmbda}')
			break

		lmbda_old = lmbda
	return lmbda


eig_max = powermethod(A, rhs)

def cheb(A, rhs, x):
	global eig_max
	eig_min  = 0.06 * eig_max
	# eig_max  = 1.2 * eig_max
	eig_avg  = (eig_min + eig_max)/2
	eig_diff = (eig_min - eig_max)/2

	# Iteration 0
	alpha = 1/eig_avg

	if x is None:
		r = -rhs
		x = np.zeros(r.shape)
	else:
		r = A * x - rhs # MV here
	p = -r
	x += alpha * p

	# Iteration 1
	r += alpha * (A * p)  # MV here
	dea = (eig_diff * alpha)
	beta = 0.5 * dea * dea
	alpha = 1/(eig_avg - (beta/alpha))
	p = -r + beta * p

	x += alpha * p

	# Iteration i>=2
	for i in range(2, 3):
		r += alpha * (A * p)  # MV here

		dea = (eig_diff * alpha)
		beta = 0.25 * dea * dea
		alpha = 1/(eig_avg - (beta/alpha))
		p = -r + beta * p

		x += alpha * p
	return x

iters = 0
def report_callback(x):
	global mat
	global rhs
	global iters

	r = rhs - mat * x
	r_norm = np.linalg.norm(r)
	iters += 1
	print(f'{iters}) {r_norm}')

maxiter = 2000
M = sp.sparse.linalg.LinearOperator((rows,rows), matvec=lambda x: cheb(A, x, None))	
x, exit_code = sp.sparse.linalg.cg(A, rhs, x0=x, M=M,maxiter=maxiter, atol=0.0, callback=report_callback)
# x, exit_code = sp.sparse.linalg.cg(A, rhs, x0=x, maxiter=maxiter*2, atol=0.0, callback=report_callback)

r = rhs - mat * x
print(f'CG({exit_code}): {np.sqrt(np.sum(r*r))}')

cheb_ctd = False
if cheb_ctd:
	# x = x * 0
	# r = rhs - A * x
	# x += r

	for i in range(0, maxiter):
		x = cheb(A, rhs, x)
		r = rhs - A * x
		print(f'CHEB: {i}) {np.sqrt(np.sum(r*r))}')
	