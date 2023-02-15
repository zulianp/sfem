#!/usr/bin/env python3

import netCDF4
import numpy as np
import sys
import os

if len(sys.argv) < 3:
	print(f'usage: {sys.argv[0]} <input_mesh> <output_folder>')
	exit()

input_mesh = sys.argv[1]
output_folder = sys.argv[2]

if not os.path.exists(output_folder):
	os.makedirs(output_folder)

nc = netCDF4.Dataset(input_mesh)
connect = nc.variables['connect1']

# print(connect[:,0])

nelements, nnodesxelem = connect.shape

for i in range(0, nnodesxelem):
	ii = np.array(connect[:,0]).astype(np.int32)
	ii.tofile(f'{output_folder}/i{i}.raw')

if 'coord' in nc.variables:
	coords = nc.variables['coord']
else:
	coords = []
	if 'coordx' in nc.variables:
		coordx = nc.variables['coordx']
		coords.append(coordx)
	if 'coordy' in nc.variables:
		coordy = nc.variables['coordy']
		coords.append(coordy)
	if 'coordz' in nc.variables:
		coordz = nc.variables['coordz']
		coords.append(coordz)

	coords = np.array(coords)
	print(coords[0])

dims, nnodes = coords.shape

coordnames = ['x', 'y', 'z', 't' ]

for i in range(0, dims):
	x = np.array(coords[i, :]).astype(np.float32)
	x.tofile(f'{output_folder}/{coordnames[i]}.raw')

# num_sidesets = nc.dimensions['num_side_sets'].size
# print(num_sidesets)

# for i in range(0, num_sidesets):
# 	ssidx = i + 1
# 	key = f'side_ss{ssidx}'
# 	ss = nc.variables[key]
# 	print(ss[:])

