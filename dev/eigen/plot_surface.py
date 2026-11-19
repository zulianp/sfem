#!/usr/bin/env python3

import numpy as np
import scipy as sp
import sys
import os
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

import glob

idx_t = np.int32
real_t = np.float64


def rf(val):
	return np.round(val, 3)

def main(argv):
	if len(argv) < 4:
		print(f'usage: {argv[0]} <eigen_pattern> <nx> <ny> [max_eigs] [output_path]')
		exit(1)

	eigen_pattern = argv[1]
	nx = int(argv[2])
	ny = int(argv[3])

	n=-1

	if(len(argv) > 4):
		n = int(argv[4])

	output_path='eigs'
	if(len(argv) > 5):
		output_path = argv[5]

	file_eigs = glob.glob(eigen_pattern, recursive=False)

	
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	X = np.arange(0, nx)
	Y = np.arange(0, ny)
	X, Y = np.meshgrid(X, Y)
	X = X.T
	Y = Y.T

	max_val = -100000000.
	min_val = 100000000.

	legend = []
	i=0
	for f in file_eigs:
		v = np.fromfile(f, dtype=real_t)
		ax.plot_surface(X, Y, np.reshape(v, (nx, ny)), cmap=cm.coolwarm)
		legend.append(os.path.basename(f))

		max_val = max(max_val, np.max(v))
		min_val = min(min_val, np.min(v))

		i += 1
		if i == n:
			break	

	plt.title(f'Vectors {eigen_pattern} [{rf(min_val)}, {rf(max_val)}]')
	fig.tight_layout()
	plt.savefig(f'{output_path}', dpi=300)

if __name__ == '__main__':
	main(sys.argv)
