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

def rf(val):
	return np.round(val, 2)

def main(argv):
	if len(argv) < 3:
		print(f'usage: {argv[0]} <crs_folder> <which=LR|SR> [output_folder]')
		exit(1)

	output_folder='eigs'
	if(len(argv) > 3):
		output_folder = argv[3]

	if not os.path.exists(output_folder):
		os.mkdir(f'{output_folder}')
	
	folder = argv[1]
	which = argv[2]

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
	diff = np.sum(np.sum(np.abs(A - A.T)))
	print()

	if diff < 1e-16:
		vals, vecs = la.eigh(A)
	else:
		vals, vecs = la.eig(A)

	

	# mag = vals * vals
	# mag = mag.real * np.sign(vals.real)
	mag = np.absolute(vals).real
	# print(mag)
	reorder = True
	if which == 'SR':
		order = np.argsort(vals.real)
	elif which == 'LR':
		order = np.argsort(-vals.real)
	elif which == 'LM':
		order = np.argsort(-mag)
		print(f'min(mag)={np.min(mag)}')
		print(f'max(mag)={np.max(mag)}')
	elif which == 'SM':
		print(f'min(mag)={np.min(mag)}')
		print(f'max(mag)={np.max(mag)}')
		order = np.argsort(mag)
	elif which == 'angle':
		angles = np.arctan2(vals.imag, vals.real)
		# print(rf(vals.real))
		# print(rf(vals.imag))

		isneg = angles < 0
		angles[isneg] = np.mod(angles[isneg] + 2*np.pi, 2*np.pi)
		# print(rf(angles))
		order = np.argsort(angles)
	else:
		reorder = False

	verbose=False
	if verbose:

		print(folder)

		print("matrix")
		print(A)
		print('values')
		print(rf(vals.real))
		print(rf(vals.imag))

		print('vecs')
		print('real')
		for k in range(0, N):
			print(rf(vecs[:,k].real))

		print('imag')
		for k in range(0, N):
			print(rf(vecs[:,k].imag))

	if reorder:
		vals = vals[order]
		vecs = vecs[:, order]

	plt.figure(figsize=(1000, 1000))
	fig, axs = plt.subplots(2, 2)

	axs[0,0].plot(vals.real)
	axs[0,0].set_title('Real')
	axs[0, 1].plot(vals.imag)
	axs[0, 1].set_title('Imag')

	axs[1, 0].plot(np.absolute(vals).real)
	axs[1, 0].set_title('Mag')

	lr = axs[1,1].plot(vals.real)
	li = axs[1,1].plot(vals.imag)
	lm = axs[1,1].plot(np.absolute(vals).real)
	axs[1,1].set_title('Both')
	axs[1,1].legend(['Real', 'Imag', 'Mag'], loc='center left', bbox_to_anchor=(1, 0.5))
	

	# plt.plot(vals.imag)
	fig.suptitle(f'Eigenvalues {which}')
	plt.xlabel('Number')
	plt.ylabel('Value')

	fig.tight_layout()
	plt.savefig(f'{output_folder}/eigvals.png', dpi=300)

	min_val = vecs[0, 0]
	max_val = vecs[0, 0]

	for k in range(0, N):
		postfix='{num:05d}.raw'.format(num=k)
		name_real = f'{output_folder}/real.{postfix}'
		name_imag = f'{output_folder}/imag.{postfix}'
		vecs[:, k].real.tofile(name_real)
		vecs[:, k].imag.tofile(name_imag)

		vr = vecs[:, k]
		minv = np.min(vr)
		maxv = np.max(vr)
		# print(f'val({k}) r={rf(vals[k].real)} i={rf(vals[k].imag)}, minv={rf(minv.real)},{rf(minv.imag)} maxv={rf(maxv.real)},{rf(maxv.imag)}')
		# print(rf(vr.real))
		# print(rf(vr.imag))

		min_val = min(min_val, minv)
		max_val = max(max_val, maxv)

	# print(f'min_val {min_val}')
	# print(f'max_val {max_val}')

	O = np.zeros((N, N), dtype=real_t)
	check_orthonormal=True
	if check_orthonormal:
		for k1 in range(0, N):
			for k2 in range(0, N):
				v1 = vecs[:, k1]
				v2 = vecs[:, k2]
				v_mag = np.sum(v1 * np.conjugate(v2))
				O[k1, k2] = v_mag.real
	
	d = np.diag(O)
	sd = np.sum(np.abs(d))
	sm = np.sum(np.sum(np.abs(O)))

	for k in range(0, N):
			O[k,k] = 0 

	plt.clf()

	ax = plt.imshow(np.abs(O))
	plt.colorbar()
	plt.savefig(f'{output_folder}/ortho.png', dpi=300)


	print(f'Orthonormal: {sd} - {sm} = {sd - sm}')



if __name__ == '__main__':
	main(sys.argv)
