#!/usr/bin/env python3

import glob
import numpy as np

try: geom_t
except NameError: 
    print('smesh: self contained mode')
    geom_t = np.float32
    idx_t  = np.int32
    real_t = np.float64
    count_t = np.int32
    elem_idx_t = np.int32

class ElementDualGraph:
	def read(self, rowptr_path, idx_path):
		self.row_ptr = np.fromfile(rowptr_path, dtype=count_t)
		self.idx 	 = np.fromfile(idx_path, dtype=elem_idx_t)

	def describe(self):
		print(self.row_ptr)
		print(self.idx)

class Block:
	def __init__(self, data):
		self.data = data
		self.cell_type = 'triangle'

class Mesh:
	def __init__(self):
		self.cells = []

	def describe(self):
		print(self.cells[0].data)
		print(self.points)
		
	def read(self, path):
		self.cells = []

		element_files = glob.glob(f'{path}/i*.raw', recursive=False)
		x_file = glob.glob(f'{path}/x.raw', recursive=False)
		y_file = glob.glob(f'{path}/y.raw', recursive=False)
		z_file = glob.glob(f'{path}/z.raw', recursive=False)

		if len(x_file) == 0 or len(y_file) == 0 or len(z_file) == 0:
			print(f'Bad coordinates in xyz files at {path}, len(xyz) = ({len(x_file)}, {len(y_file)}, ({len(z_file)})\n')
			exit(1)

		point_files = [x_file[0], y_file[0]]

		if len(z_file) > 0:
			point_files.append(z_file[0])

		list_of_idx = []
		for ef in element_files:
			idx_i = np.fromfile(ef, dtype=idx_t)
			list_of_idx.append(idx_i)

		data = np.zeros((len(list_of_idx[0]), len(list_of_idx)))

		d = 0
		for li in list_of_idx:
			data[:,d] = li[:]
			d += 1

		block = Block(data)
		self.cells.append(block)

		list_of_points = []
		for pf in point_files:
			list_of_points.append(np.fromfile(pf, dtype=geom_t))

		d = 0
		coords = np.zeros((len(list_of_points[0]), len(list_of_points)))
		for lp in list_of_points:
			coords[:,d] = lp[:]
			d += 1

		self.points = coords

	def to_meshio(self):
		import meshio
		cells = []

		for b in self.cells:
			cells.append((b.cell_type, b.data))
		return meshio.Mesh(self.points, cells)

	def write(self, path):
		m = self.to_meshio()
		m.write(path)

	def add_cells(self, cells):
		block = Block(cells)
		self.cells.append(block)

def read(input_path):
	mesh = Mesh()
	mesh.read(input_path)
	return mesh

def read_dual_graph(input_path):
	dual = ElementDualGraph()
	dual.read(f'{input_path}/adj_ptr.raw', f'{input_path}/adj_idx.raw')
	return dual

if __name__ == '__main__':
	import sys

	mesh = read(sys.argv[1])
	mesh.describe()

	if len(sys.argv) > 3:
		dual = ElementDualGraph()
		dual.read(sys.argv[2], sys.argv[3])
		dual.describe()
