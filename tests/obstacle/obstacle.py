#!/usr/bin/env python3

import pysfem as sfem
import numpy as np
from numpy import linalg
import sfem.mesh.rectangle_mesh as rectangle_mesh
import sfem.mesh.box_mesh as box_mesh

import sys, getopt, os
# import pdb

idx_t = np.int32
real_t = np.float64

def solve_obstacle(options):
	path = options.input_mesh

	if not os.path.exists(options.output_dir):
		os.mkdir(f'{options.output_dir}')

	n = 4
	h = 1./(n - 1)

	if path == "gen:rectangle":
		idx, points = rectangle_mesh.create(2, 1, 2*n, n, "triangle")

		select_inlet  = np.abs(points[0]) 	< 1e-8
		select_outlet = np.abs(points[0] - 2) < 1e-8
		select_walls  = np.logical_or(np.abs(points[1]) < 1e-8, np.abs(points[1] - 1) < 1e-8)

		sinlet  = np.array(np.where(select_inlet), dtype=idx_t)
		soutlet = np.array(np.where(select_outlet), dtype=idx_t)
		swalls  = np.array(np.where(select_walls), dtype=idx_t)

		m = sfem.create_mesh("TRI3", np.array(idx), np.array(points))
		m.write(f"{options.output_dir}/rect_mesh")
	elif path == "gen:box":
		idx, points = box_mesh.create(2, 1, 1, n * 2, n * 1, n * 1, "tet4")
		
		select_inlet  = np.abs(points[0]) 	< 1e-8
		select_outlet = np.abs(points[0] - 2) < 1e-8
		select_walls  = np.logical_or(
			np.logical_or(
				np.abs(points[1]) < 1e-8, 
				np.abs(points[1] - 1) < 1e-8),
			np.logical_or(
				np.abs(points[2]) < 1e-8,
				np.abs(points[2] - 1) < 1e-8))

		sinlet  = np.array(np.where(select_inlet), dtype=idx_t)
		soutlet = np.array(np.where(select_outlet), dtype=idx_t)
		swalls  = np.array(np.where(select_walls), dtype=idx_t)

		m = sfem.create_mesh("TET4", np.array(idx), np.array(points))
		# m = sfem.create_mesh("HEX8", np.array(idx), np.array(points))
		m.write(f"{options.output_dir}/rect_mesh")
		# pdb.set_trace()
	else:
		m = sfem.Mesh()		
		m.read(path)
		# m.convert_to_macro_element_mesh()
		sinlet = np.unique(np.fromfile(f'{path}/sidesets_aos/sinlet.raw', dtype=idx_t))
		soutlet = np.fromfile(f'{path}/sidesets_aos/soutlet.raw', dtype=idx_t)


	fs = sfem.FunctionSpace(m, m.spatial_dimension())
	fun = sfem.Function(fs)
	
	fun.set_output_dir(options.output_dir)
	out = fun.output()

	linear_elasticity_op = sfem.create_op(fs, "LinearElasticity")
	fun.add_operator(linear_elasticity_op)

	bc = sfem.DirichletConditions(fs)
	sfem.add_condition(bc, sinlet, 0, 0.2);
	sfem.add_condition(bc, sinlet, 1, 0.);
	sfem.add_condition(bc, sinlet, 2, 0.);
	fun.add_dirichlet_conditions(bc)

	cg = sfem.ConjugateGradient()
	cg.default_init()
	cg.set_max_it(20)
	cg.set_rtol(1e-2)
	cg.set_atol(1e-12)
	cg.set_verbose(False)

	g = np.zeros(fs.n_dofs())
	c = np.zeros(fs.n_dofs())
	x = np.zeros(fs.n_dofs())
	sfem.apply_constraints(fun, x)

	mass_op = sfem.create_op(fs, "LumpedMass")
	mass = np.zeros(fs.n_dofs(), dtype=real_t)
	sfem.hessian_diag(mass_op, x, mass)

	obs = np.zeros(fs.n_dofs()) + 10000

	constrained_dofs = soutlet[:]*m.spatial_dimension()

	sdf = 0.6 - sfem.points(m, 0)
	obs[constrained_dofs] = sdf[soutlet]

	for d in range(1, m.spatial_dimension()):
		obs[d::m.spatial_dimension()] = 10000

	penalty_param = 1e2

	# Nonlinear iteration
	for i in range(0, 500):
		g[:] = 0
			
		violation = np.maximum(x - obs, 0)
		H_diag = penalty_param * mass * ((obs - x) <= 0)
		sfem.apply_zero_constraints(fun, H_diag)
		H = sfem.diag(H_diag)
		
		g = penalty_param * mass * violation
		sfem.apply_zero_constraints(fun, g)
		sfem.gradient(fun, x, g)
		
		
		# As the pointer of x is used not necessary to be in the loop
		# But maybe there will be some side effect in the future
		lop = H + sfem.make_op(fun, x)
		cg.set_op(lop)

		c[:] = 0
		sfem.apply(cg, g, c)
		x -= c

		norm_g = linalg.norm(g)
		print(f'{i}) norm_g = {norm_g}')
		if(norm_g < 1e-10):
			break

	sfem.write(out, "disp", x)

class Opts:
	def __init__(self):
		self.input_mesh = ''
		self.output_dir = './output'

if __name__ == '__main__':
	print(sys.argv)
	if len(sys.argv) < 3:
		print(f'usage: {sys.argv[0]} <input_mesh> <output>')
		exit(1)

	sfem.init()

	options = Opts()
	options.input_mesh = sys.argv[1]
	options.output = sys.argv[2]
	
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

	solve_obstacle(options)
	sfem.finalize()
