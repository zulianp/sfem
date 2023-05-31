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

	legend = []
	i=0
	for f in file_eigs:
		v = np.fromfile(f, dtype=real_t)
		# plt.plot(np.tile(v, 4))
		# ax.plot_surface(X, Y, np.reshape(v, (nx, ny)), cmap=cm.coolwarm)
		ax.plot_surface(X, Y, np.reshape(v, (nx, ny)), cmap=cm.coolwarm)
		legend.append(os.path.basename(f))

		i += 1
		if i == n:
			break	


	# plt.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(f'Vectors {eigen_pattern}')
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
