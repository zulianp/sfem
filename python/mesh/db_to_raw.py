#!/usr/bin/env python3

import meshio
import numpy as np
import sys
import os

def main(argv):
	if len(argv) < 3:
		print(f'usage: {argv[0]} <input_mesh> <output_folder>')
		exit()

	input_mesh = argv[1]
	output_folder = argv[2]

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	mesh = meshio.read(input_mesh)

	###################################
	# Export indices
	###################################

	nblocks = len(mesh.cells)
	if nblocks > 1:
		print(f'nblocks={nblocks}')

	for b in mesh.cells:
		ncells, nnodesxelem = b.data.shape

		for d in range(0, nnodesxelem):
			i0 = b.data[:, d]
			i0.astype(np.int32).tofile(f'{output_folder}/i{d}.raw')

	###################################
	# Points
	###################################

	xyz = np.transpose(mesh.points)
	ndims, nodes = xyz.shape
	str_xyz = ['x', 'y', 'z', 't']

	for d in range(0, ndims):
		x = xyz[d, :].astype(np.float32)
		x.tofile(f'{output_folder}/{str_xyz[d]}.raw')


	###################################
	# Point data
	###################################

	point_data_dir = f'{output_folder}/point_data/'

	if not os.path.exists(point_data_dir):
		os.makedirs(point_data_dir)

	print("Point data:")
	for key in mesh.point_data:
		print(f"\t- {key}")
		data = mesh.point_data[key]
		d = data[:].astype(np.float64)
		d.tofile(f'{point_data_dir}/{key}.raw')

	###################################
	# Cell data
	###################################

	cell_data = f'{output_folder}/cell_data/'

	if not os.path.exists(cell_data):
		os.makedirs(cell_data)

	print("Cell data:")
	for key in mesh.cell_data:
		print(f"\t- {key}")
		data = mesh.cell_data[key]
		d = data[:].astype(np.float64)
		d.tofile(f'{cell_data}/{key}.raw')

if __name__ == '__main__':
    main(sys.argv)
