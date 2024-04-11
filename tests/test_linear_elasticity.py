#!/usr/bin/env python3

import pysfem as sfem
import numpy as np
from numpy import linalg

import sys, getopt

idx_t = np.int32

def solve_linear_elasticity(options):
	sfem.init()
	m = sfem.Mesh()

	path = options.input_mesh
	m.read(path)
	# m.convert_to_macro_element_mesh()

	sinlet = np.fromfile(f'{path}/sidesets_aos/sinlet.raw', dtype=idx_t)
	soutlet = np.fromfile(f'{path}/sidesets_aos/soutlet.raw', dtype=idx_t)

	fs = sfem.FunctionSpace(m, m.spatial_dimension())
	fun = sfem.Function(fs)
	fun.set_output_dir(options.output_dir)

	op = sfem.create_op(fs, "LinearElasticity")
	fun.add_operator(op)

	bc = sfem.DirichletConditions(fs)

	sfem.add_condition(bc, sinlet, 0, 0.);
	sfem.add_condition(bc, sinlet, 1, 0.);
	sfem.add_condition(bc, sinlet, 2, 0.);

	# sfem.add_condition(bc, soutlet, 0, 1);
	# sfem.add_condition(bc, soutlet, 1, 0.);
	# sfem.add_condition(bc, soutlet, 2, 0.);

	fun.add_dirichlet_conditions(bc)

	nc = sfem.NeumannConditions(fs)
	sfem.add_condition(nc, soutlet, 0, 1)
	fun.add_operator(nc)

	x = np.zeros(fs.n_dofs())
	cg = sfem.ConjugateGradient()
	cg.default_init()
	cg.set_max_it(3000)

	lop = sfem.make_op(fun, x)
	cg.set_op(lop)
	
	g = np.zeros(fs.n_dofs())
	sfem.apply_constraints(fun, x)
	sfem.gradient(fun, x, g)
	c = np.zeros(fs.n_dofs())
	sfem.apply(cg, g, c)
	x -= c

	# Write result to disk
	sfem.report_solution(fun, x)

class Opts:
	def __init__(self):
		self.input_mesh = ''
		self.output_dir = './output'

if __name__ == '__main__':
	print(sys.argv)
	if len(sys.argv) < 3:
		print(f'usage: {sys.argv[0]} <input_mesh> <output>')
		exit(1)

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

	solve_linear_elasticity(options)
