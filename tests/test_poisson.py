#!/usr/bin/env python3


import numpy as np
from numpy import linalg
import sfem.mesh.rectangle_mesh as rectangle_mesh
import sfem.mesh.box_mesh as box_mesh
import pysfem as pysfem
import sys, getopt
import pdb
import os

idx_t = np.int32

def gradient_descent(fun, x):
	g = np.zeros(fs.n_dofs())

	alpha = 0.1
	max_it = 1
	for k in range(0, max_it):
		# Reset content of g to zero before calling gradient
		g.fill(0)
		pysfem.gradient(fun, x, g)
		
		x -= alpha * g

		norm_g = linalg.norm(g)
		stop = norm_g < 1e-5
		if np.mod(k, 1000) == 0 or stop:
			val = pysfem.value(fun, x)
			print(f'{k}) v = {val}, norm(g) = {norm_g}')
			
		if stop:
			break
	return x

def solve_poisson(options):

	path = options.input_mesh

	if not os.path.exists(options.output_dir):
		os.mkdir(f'{options.output_dir}')

	if path == "gen:rectangle":
		idx, points = rectangle_mesh.create(2, 1, 200, 100, "triangle")
		sinlet  = np.array(np.where(np.abs(points[0]) 	< 1e-8), dtype=idx_t)
		soutlet = np.array(np.where(np.abs(points[0] - 2) < 1e-8), dtype=idx_t)
		m = pysfem.create_mesh("TRI3", np.array(idx), np.array(points))
		m.write(f"{options.output_dir}/rect_mesh")
	elif path == "gen:box":
		resolution = 10
		idx, points = box_mesh.create(2, 1, 1, resolution * 20, resolution * 10, resolution * 10, "tet4")
		sinlet  = np.array(np.where(np.abs(points[0]) 	< 1e-8), dtype=idx_t)
		soutlet = np.array(np.where(np.abs(points[0] - 2) < 1e-8), dtype=idx_t)
		m = pysfem.create_mesh("TET4", np.array(idx), np.array(points))
		m.write(f"{options.output_dir}/rect_mesh")
	else:
		m = pysfem.Mesh()		
		m.read(path)
		m.convert_to_macro_element_mesh()
		sinlet = np.unique(np.fromfile(f'{path}/sidesets_aos/sinlet.raw', dtype=idx_t))
		soutlet = np.fromfile(f'{path}/sidesets_aos/soutlet.raw', dtype=idx_t)

	fs  = pysfem.FunctionSpace(m, 1)
	fun = pysfem.Function(fs)
	fun.set_output_dir(options.output_dir)

	op = pysfem.create_op(fs, "Laplacian")
	fun.add_operator(op)

	bc = pysfem.DirichletConditions(fs)
	pysfem.add_condition(bc, sinlet, 0, -1);
	pysfem.add_condition(bc, soutlet, 0, 1);
	fun.add_dirichlet_conditions(bc)
	fun.set_output_dir(options.output_dir)

	# If we have surface element sideset we can use Neumann conditions
	# nc = pysfem.NeumannConditions(fs)
	# pysfem.add_condition(nc, soutlet, 0, 1)
	# fun.add_operator(nc)

	x = np.zeros(fs.n_dofs())
	g = np.zeros(fs.n_dofs())
	c = np.zeros(fs.n_dofs())
	
	cg = pysfem.ConjugateGradient()
	cg.default_init()
	cg.set_max_it(30000)

	lop = pysfem.make_op(fun, x)
	cg.set_op(lop)

	pysfem.apply_constraints(fun, x)
	pysfem.gradient(fun, x, g)
	pysfem.apply(cg, g, c)
	
	# Apply correction 
	x -= c

	pysfem.report_solution(fun, x)

class Opts:
	def __init__(self):
		self.input_mesh = ''
		self.output_dir = '.'

if __name__ == '__main__':
	print(sys.argv)
	if len(sys.argv) < 3:
		print(f'usage: {sys.argv[0]} <input_mesh> <output.raw>')
		exit(1)

	options = Opts()
	options.input_mesh = sys.argv[1]
	options.output_dir = sys.argv[2]
	
	try:
	    opts, args = getopt.getopt(
	        sys.argv[3:], "h",
	        ["help"])

	except getopt.GetoptError as err:
	    print(err)
	    print(usage)
	    sys.exit(1)

	for opt, arg in opts:
	    if opt in ('-h', '--help'):
	        print(usage)
	        sys.exit()

	pysfem.init()
	solve_poisson(options)
	pysfem.finalize()
