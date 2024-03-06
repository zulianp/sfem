#!/usr/bin/env python3

import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

real_t = np.float64
complex_t = np.cdouble


def rf(val):
	return np.round(val, 4)

def main(argv):
	if len(argv) < 5:
		print(f'usage {argv[0]} <path_basis> <num_vecs> <thresh> <data.raw>')

	path_basis = argv[1]
	num_vecs = int(argv[2])
	thresh= float(argv[3])
	path_data = argv[4]

	data_real_t = np.float64
	if len(argv) > 5:
		data_real_t = argv[5]

	files_basis = glob.glob(path_basis, recursive=False)
	files_data  = glob.glob(path_data,  recursive=False)

	if len(files_data) == 0:
		exit(1)

	print(files_data)

	basis = []

	count_basis = 0
	for f in files_basis:
		re = np.fromfile(f, dtype=real_t)
		im = np.fromfile(f.replace("real", "imag"), dtype=real_t)
		d = re.astype(dtype=complex_t)
		d.imag = im

		basis.append(d)
		count_basis += 1
		if count_basis == num_vecs:
			break

	N = len(basis)

	print(f'Read {N} basis vectors' )

	check_orthonormal=True
	if check_orthonormal:
		O = np.zeros((N, N), dtype=real_t)
		for k1 in range(0, N):
			for k2 in range(0, N):
				v1 =basis[k1]
				v2 =basis[k2]
				v_mag = np.sum(v1 * np.conjugate(v2))
				O[k1, k2] = np.absolute(v_mag).real
	
		d = np.diag(O)
		sd = np.sum(np.abs(d))
		sm = np.sum(np.sum(np.abs(O)))

		# for k in range(0, N):
		# 		O[k,k] = 0 

		plt.clf()
		ax = plt.imshow(np.abs(O))
		plt.colorbar()
		plt.savefig(f'./project_ortho.png', dpi=300)
		print(f'Orthonormal: {sd} - {sm} = {sd - sm}')


	if not os.path.exists('reconstructed'):
		os.mkdir('reconstructed')

	for f in files_data:
		d = np.fromfile(f, dtype=data_real_t).astype(real_t)
		x = np.zeros(d.shape, dtype=complex_t)

		ampl = np.zeros(N, dtype=complex_t)
		for i in range(0, N):
			ampl[i] = np.sum(np.conjugate(basis[i]) * d)
			print(f'{rf(ampl[i].real)}, {rf(ampl[i].imag)}')



		max_amp = np.max(np.absolute(ampl))

		nskip = 0
		for i in range(0, N):
			if np.absolute(ampl[i]/max_amp).real > thresh:
				# x += np.conjugate(ampl[i]) * basis[i]
				x += ampl[i] * basis[i]
				
			else:
				nskip += 1

		print(f'nskip = {nskip}')

		# print(rf(x.real))
		# print(rf(x.imag))

		# print(f'{np.max(np.abs(x.real))}')
		# print(f'{np.max(np.abs(x.imag))}')


		np.real(x).tofile(f'reconstructed/r{os.path.basename(f)}')
		np.imag(x).tofile(f'reconstructed/i{os.path.basename(f)}')
		np.absolute(x).tofile(f'reconstructed/m{os.path.basename(f)}')

if __name__ == '__main__':
	main(sys.argv)
