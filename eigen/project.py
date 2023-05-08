#!/usr/bin/env python3

import glob
import os
import sys
import numpy as np

real_t = np.float64

def main(argv):
	if len(argv) < 4:
		print(f'usage {argv[0]} <path_basis> <num_vecs> <data.raw>')

	path_basis = argv[1]
	num_vecs = int(argv[2])
	path_data = argv[3]

	data_real_t = np.float64
	if len(argv) > 4:
		data_real_t = argv[4]

	files_basis = glob.glob(path_basis, recursive=False)
	files_data  = glob.glob(path_data,  recursive=False)

	if len(files_data) == 0:
		exit(1)

	print(files_data)

	basis = []

	count_basis = 0
	for f in files_basis:
		d = np.fromfile(f, dtype=real_t)
		basis.append(d)

		if count_basis == num_vecs:
			break

	N = len(basis)

	if not os.path.exists('reconstructed'):
		os.mkdir('reconstructed')

	for f in files_data:
		d = np.fromfile(f, dtype=data_real_t).astype(real_t)
		x = np.zeros(d.shape, dtype=real_t)

		ampl = np.zeros(N, dtype=real_t)
		for i in range(0, N):
			ampl[i] = np.sum(basis[i] * d)

		for i in range(0, N):
			x += ampl[i] * basis[i]

		x.tofile(f'reconstructed/r{os.path.basename(f)}')

if __name__ == '__main__':
	main(sys.argv)
