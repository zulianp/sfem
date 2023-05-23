#!/usr/bin/env python3

import netCDF4
import numpy as np
import sys
import os

import pdb

geom_type = np.float32
idx_type = np.int32

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
	t = np.array(time_whole[:]).astype(np.float32)
	t.tofile(f'{output_folder}/time_whole.raw')

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
		
def s2n_quad4():
	return [
		[0, 1],
		[1, 2],
		[2, 3],
		[3, 0],
	]

def s2n_hex8():
	return [
		[0, 1, 5, 4],
		[1, 2, 6, 5],
		[2, 3, 7, 6],
		[3, 0, 4, 7],
		[0, 1, 2, 3],
		[4, 5, 6, 7],
	]

def s2n_tet4():
	return [
		[0, 1, 3],
		[1, 2, 3],
		[0, 3, 2],
		[0, 1, 2],
	]

def s2n_tri3():
	return [
		[0, 1],
		[1, 2],
		[2, 0],
	]

ss_to_nodelist = {}
ss_to_nodelist['QUAD4'] = s2n_quad4()
ss_to_nodelist['HEX8'] = s2n_hex8()
ss_to_nodelist['TET4'] = s2n_tet4()
ss_to_nodelist['TETRA'] = s2n_tet4()
ss_to_nodelist['tetra'] = s2n_tet4()
ss_to_nodelist['TRI3'] = s2n_tri3()

#########################################
# Elements
#########################################


num_elem = nc.dimensions['num_elem'].size
print(f'num_elem = {num_elem}')

num_el_blk = nc.dimensions['num_el_blk'].size
print(f'num_el_blk = {num_el_blk}')

if num_el_blk == 1:
	connect = nc.variables['connect1']
	elem_type = connect.elem_type
	print(f'elem_type = {elem_type}')

	nelements, nnodesxelem = connect.shape

	for i in range(0, nnodesxelem):
		ii = np.array(connect[:,i]).astype(idx_type) - 1
		ii.tofile(f'{output_folder}/i{i}.raw')

else:
	num_nod_per_el_ref = 0
	for b in range(0, num_el_blk):
		var_name = f'num_nod_per_el{b+1}'
		num_nod_per_el = nc.dimensions[var_name].size
		print(f'{var_name} = {num_nod_per_el}')

		if num_nod_per_el_ref == 0:
			num_nod_per_el_ref = num_nod_per_el
		else:
			assert num_nod_per_el_ref == num_nod_per_el

	connect = np.zeros((num_elem, num_nod_per_el_ref), dtype=idx_type)

	offset = 0
	for b in range(0, num_el_blk):
		connect_b = nc.variables[f'connect{b+1}']
		elem_type = connect_b.elem_type
		print(f'elem_type = {elem_type}')

		nelements, nnodesxelem = connect_b.shape

		connect[offset:(offset + nelements),:] = connect_b[:].astype(idx_type)
		offset += nelements

	for i in range(0, nnodesxelem):
		ii = np.array(connect[:,i]).astype(idx_type) - 1
		ii.tofile(f'{output_folder}/i{i}.raw')

#########################################
# Sidesets
#########################################

num_sidesets = nc.dimensions['num_side_sets'].size
print(f'num_sidesets={num_sidesets}')

s2n_map = ss_to_nodelist[elem_type]
nnodesxside = len(s2n_map[0])

ss_names = nc.variables['ss_names']

sideset_dir = f'{output_folder}/sidesets'
if num_sidesets > 0:
	mkdir(sideset_dir)

for i in range(0, num_sidesets):
	ssidx = i + 1

	name = netCDF4.chartostring(ss_names[i])

	if name == "":
		name = f"sideset{ssidx}"

	print(f'sideset = {name}')

	key = f'elem_ss{ssidx}'
	e_ss = nc.variables[key]

	key = f'side_ss{ssidx}'
	s_ss = nc.variables[key]

	this_sideset_dir = f'{sideset_dir}/{name}'
	mkdir(this_sideset_dir)

	idx = [None] * nnodesxside
	for d in range(0, nnodesxside):
		idx[d] = []

	for n in range(0, len(e_ss[:])):
		e = e_ss[n] - 1
		s = s_ss[n] - 1

		lnodes = s2n_map[s]

		for d in range(0, nnodesxside):
			ln = lnodes[d]
			node = connect[e, ln] - 1

			# if(node == 162):
			# 	pdb.set_trace()

			idx[d].append(node)

	for d in range(0, nnodesxside):
		path = f'{this_sideset_dir}/{name}.{d}.raw'
		ii = np.array(idx[d]).astype(idx_type)
		ii.tofile(path)

