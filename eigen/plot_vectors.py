#!/usr/bin/env python3

import numpy as np
import scipy as sp
import sys
import os
import math
import matplotlib.pyplot as plt
import glob

idx_t = np.int32
real_t = np.float64


def main(argv):
	if len(argv) < 2:
		print(f'usage: {argv[0]} <eigen_pattern> [max_eigs] [output_path]')
		exit(1)

	eigen_pattern = argv[1]
	n=-1

	if(len(argv) > 2):
		n = int(argv[2])

	output_path='eigs'
	if(len(argv) > 3):
		output_path = argv[3]

	file_eigs = glob.glob(eigen_pattern, recursive=False)

	i=0
	for f in file_eigs:
		v = np.fromfile(f, dtype=real_t)
		plt.plot(np.tile(v, 4))

		i += 1
		if i == n:
			break

	plt.title(f'Vectors {eigen_pattern}')
	plt.savefig(f'{output_path}', dpi=300)

if __name__ == '__main__':
	main(sys.argv)
