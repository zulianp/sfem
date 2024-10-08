#!/usr/bin/env python3

import meshio
import numpy as np
import sys, getopt
import os
import glob
import pdb

idx_t = np.int32
geom_t = np.float32

if __name__ == '__main__':
	argv = sys.argv
	usage = f'usage: {argv[0]} <level> <input_folder> <output_folder>'

	if(len(argv) < 4):
	    print(usage)
	    sys.exit(1)

	L = int(argv[1])
	input_folder = argv[2]
	output_folder = argv[3]

	nnodesxelem = (L+1) * (L+1)

	x = np.fromfile(f'{input_folder}/x.raw', dtype=geom_t)
	y = np.fromfile(f'{input_folder}/y.raw', dtype=geom_t)
	z = np.fromfile(f'{input_folder}/z.raw', dtype=geom_t)

	elements = [0]*nnodesxelem
	for d in range(0, nnodesxelem):
		elements[d] = np.fromfile(f'{input_folder}/i{d}.raw', dtype=idx_t)

	nmicroelements = L * L 

	N = len(elements[0])
	i0 = np.zeros(N * nmicroelements, dtype=idx_t)
	i1 = np.zeros(N * nmicroelements, dtype=idx_t)
	i2 = np.zeros(N * nmicroelements, dtype=idx_t)
	i3 = np.zeros(N * nmicroelements, dtype=idx_t)

	ld = [1, (L+1)]

	offset = 0
	for j in range(0, L):
		for i in range(0, L):
			i0[offset:(offset + N)] = elements[j * ld[1] + i * ld[0]]
			i1[offset:(offset + N)] = elements[j * ld[1] + (i+1) * ld[0]]
			i2[offset:(offset + N)] = elements[(j+1) * ld[1] + (i+1) * ld[0]]
			i3[offset:(offset + N)] = elements[(j+1) * ld[1] + i * ld[0]]
			offset += N

	print("STAS: ", len(x), np.max(elements[0]))

	refined_n_nodes = 0
	for d in range(0, nnodesxelem):
		refined_n_nodes = max(refined_n_nodes, np.max(elements[d]))
	refined_n_nodes += 1

	print(refined_n_nodes)

	def phi0(x, y):
		return (1 - x) * (1 - y)

	def phi1(x, y):
		return x * (1 - y)

	def phi2(x, y):
		return x * y

	def phi3(x, y):
		return (1 - x) * (y)

	# Corners
	# X
	x0 = x[elements[0]]
	x1 = x[elements[L]]
	x2 = x[elements[L * ld[1] + L]]
	x3 = x[elements[L * ld[1] + 0]]

	# Y
	y0 = y[elements[0]]
	y1 = y[elements[L]]
	y2 = y[elements[L * ld[1] + L]]
	y3 = y[elements[L * ld[1] + 0]]

	# Z
	z0 = z[elements[0]]
	z1 = z[elements[L]]
	z2 = z[elements[L * ld[1] + L]]
	z3 = z[elements[L * ld[1] + 0]]

	mx = np.zeros(refined_n_nodes, dtype=geom_t)
	my = np.zeros(refined_n_nodes, dtype=geom_t)
	mz = np.zeros(refined_n_nodes, dtype=geom_t)

	h = 1./L
	for j in range(0, L+1):
		for i in range(0, L+1):
			# print(f'{i, j}')

			f0 = phi0(i*h, j*h)
			f1 = phi1(i*h, j*h)
			f2 = phi2(i*h, j*h)
			f3 = phi3(i*h, j*h)

			xi = f0 * x0 + f1 * x1 + f2 * x2 + f3 * x3
			yi = f0 * y0 + f1 * y1 + f2 * y2 + f3 * y3
			zi = f0 * z0 + f1 * z1 + f2 * z2 + f3 * z3

			ii = elements[j * ld[1] + i * ld[0]]

			mx[ii] = xi
			my[ii] = yi
			mz[ii] = zi

	if not os.path.exists(output_folder):
	    os.mkdir(output_folder)

	i0.tofile(f'{output_folder}/i0.raw')
	i1.tofile(f'{output_folder}/i1.raw')
	i2.tofile(f'{output_folder}/i2.raw')
	i3.tofile(f'{output_folder}/i3.raw')

	mx.tofile(f'{output_folder}/x.raw')
	my.tofile(f'{output_folder}/y.raw')
	mz.tofile(f'{output_folder}/z.raw')

