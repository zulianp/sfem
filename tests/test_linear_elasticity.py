#!/usr/bin/env python3

import pysfem as sfem
import numpy as np
from numpy import linalg
from time import perf_counter

import sys, getopt

idx_t = np.int32

def solve_linear_elasticity(options):
	tick = perf_counter()

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

	# op = sfem.create_op(fs, "LinearElasticity")
	op = sfem.create_op(fs, "NeoHookeanOgden")
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
	cg.set_max_it(50)
	cg.set_rtol(1e-2)
	cg.set_atol(1e-12)
	cg.set_verbose(False)
	
	damping = 1

	# use_prec = True
	use_prec = False
	if use_prec:
		d = np.zeros(fs.n_dofs())
		sfem.hessian_diag(fun, x, d)
		prec = sfem.diag(1./d)
		cg.set_preconditioner_op(prec)
	
		print(np.max(d))
		print(np.min(d))
		print(np.max(np.abs(d)))
		print(np.min(np.abs(d)))

	# Newton iteration
	g = np.zeros(fs.n_dofs())
	c = np.zeros(fs.n_dofs())
	for i in range(0, 100):
		g[:] = 0
		sfem.gradient(fun, x, g)
		
		# As the pointer of x is used not necessary to be in the loop
		# But maybe there will be some side effect in the future
		lop = sfem.make_op(fun, x)
		cg.set_op(lop)

		c[:] = 0
		sfem.apply(cg, g, c)
		x -= damping * c

		norm_g = linalg.norm(g)
		print(f'{i}) norm_g = {norm_g}')
		if(norm_g < 1e-10):
			break

	# Write result to disk
	sfem.report_solution(fun, x)

	tock = perf_counter()

	print(f'Summary: {fs.n_dofs()} dofs, {round(tock - tick, 4)} seconds')

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
