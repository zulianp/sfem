#!/usr/bin/env python3

import netCDF4
import numpy as np
import sys
import os

import pdb

geom_type = np.float32

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

if len(sys.argv) < 3:
	print(f'usage: {sys.argv[0]} <input_mesh> <output_folder>')
	exit()

input_mesh = sys.argv[1]
output_folder = sys.argv[2]

mkdir(output_folder)

nc = netCDF4.Dataset(input_mesh)
connect = nc.variables['connect1']

nelements, nnodesxelem = connect.shape

for i in range(0, nnodesxelem):
	ii = np.array(connect[:,i]).astype(np.int32) - 1
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

dims, nnodes = coords.shape

coordnames = ['x', 'y', 'z', 't' ]

for i in range(0, dims):
	x = np.array(coords[i, :]).astype(geom_type)
	x.tofile(f'{output_folder}/{coordnames[i]}.raw')

n_time_steps = 1
if 'time_whole' in nc.variables:
	time_whole = nc.variables['time_whole']
	n_time_steps = time_whole.shape[0]

print(f'n_time_steps = {n_time_steps}')

if 'name_nod_var' in nc.variables:
	name_nod_var = nc.variables['name_nod_var']
	nvars, __ = name_nod_var.shape
	print(f'Point data, nvars = {nvars}')

	point_data_dir = f'{output_folder}/point_data'
	mkdir(point_data_dir)

	nodal_prefix = 'vals_nod_var'
	for i in range(0, nvars):
		var_key = f'{nodal_prefix}{i+1}'
		var = nc.variables[var_key]

		var_name = netCDF4.chartostring(name_nod_var[i, :])
		print(f' - {var_name}, dtype {var.dtype}')

		var_path_prefix = f'{point_data_dir}/{var_name}'

		if(n_time_steps <= 1):
			path = f'{var_path_prefix}.raw'

			data = var[t, 0]
			data.tofile(path)
		else:
			size_padding = int(np.ceil(np.log10(n_time_steps)))

			format_string = f"%s.%0.{size_padding}d.raw"

			for t in range(0, n_time_steps):
				data = np.array(var[t,:])

				path = format_string % (var_path_prefix, t)
				data.tofile(path)
		
# pdb.set_trace()

# num_sidesets = nc.dimensions['num_side_sets'].size
# print(num_sidesets)

# for i in range(0, num_sidesets):
# 	ssidx = i + 1
# 	key = f'side_ss{ssidx}'
# 	ss = nc.variables[key]
# 	print(ss[:])

