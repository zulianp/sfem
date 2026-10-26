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
	return np.round(val, 2)

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
	file_eigs.sort()


	X = np.arange(0, nx)
	Y = np.arange(0, ny)
	X, Y = np.meshgrid(X, Y)
	X = X.T
	Y = Y.T

	legend = []
	i=0

	N = len(file_eigs)
	m = int(np.sqrt(N))
	n = max(2, int(N/m))

	fig, axs = plt.subplots(n, m)

	fidx = 0
	for i in range(0, n):
		for j in range(0, m):
			if fidx >= N:
				break

			f = file_eigs[fidx]
			v = np.fromfile(f, dtype=real_t)
			
			axs[i,j].imshow(np.reshape(v, (nx, ny)), cmap=cm.coolwarm)

			name = os.path.basename(f)
			parts = name.split('.')
			axs[i,j].set_title(f'{parts[0]} [{rf(np.min(v))},{rf(np.max(v))}]')
			legend.append(os.path.basename(f))
			fidx += 1
			

	plt.suptitle(f'Vectors {eigen_pattern}')
	fig.tight_layout()
	plt.savefig(f'{output_path}', dpi=300)

if __name__ == '__main__':
	main(sys.argv)



# X = np.arange(0, nx)
# Y = np.arange(0, ny)
# X, Y = np.meshgrid(X, Y)

# plt.figure(figsize=(1000, 1000))
# fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})

# axs[0, 0].plot_surface(X, Y, np.reshape(vals.real, (nx, ny)), cmap=cm.coolwarm)
# axs[0, 0].set_title('Real')
# axs[0, 1].plot_surface(X, Y, np.reshape(vals.imag, (nx, ny)))
# axs[0, 1].set_title('Imag')

# axs[1, 0].plot_surface(X, Y, np.reshape(np.absolute(vals).real, (nx, ny)))
# axs[1, 0].set_title('Mag')

# lr = axs[1,1].plot_surface(X, Y, np.reshape(vals.real, (nx, ny)))
# li = axs[1,1].plot_surface(X, Y, np.reshape(vals.imag, (nx, ny)))
# lm = axs[1,1].plot_surface(X, Y, np.reshape(np.absolute(vals).real, (nx, ny)))
# axs[1,1].set_title('Both')
