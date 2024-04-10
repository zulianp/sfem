#!/usr/bin/env python3

import pysfem as sfem
import numpy as np
from numpy import linalg

import sys, getopt

idx_t = np.int32

def gradient_descent(fun, x):
	g = np.zeros(fs.n_dofs())

	alpha = 0.1
	max_it = 1
	for k in range(0, max_it):
		# Reset content of g to zero before calling gradient
		g.fill(0)
		sfem.gradient(fun, x, g)
		
		x -= alpha * g

		norm_g = linalg.norm(g)
		stop = norm_g < 1e-5
		if np.mod(k, 1000) == 0 or stop:
			val = sfem.value(fun, x)
			print(f'{k}) v = {val}, norm(g) = {norm_g}')
			
		if stop:
			break
	return x

def solve_poisson(options):
	sfem.init()
	m = sfem.Mesh()
	path = options.input_mesh
	m.read(path)
	m.convert_to_macro_element_mesh()

	sinlet = np.fromfile(f'{path}/sidesets_aos/sinlet.raw', dtype=idx_t)
	soutlet = np.fromfile(f'{path}/sidesets_aos/soutlet.raw', dtype=idx_t)

	fs = sfem.FunctionSpace(m, 1)
	fun = sfem.Function(fs)
	fun.set_output_dir(options.output_dir)

	op = sfem.create_op(fs, "Laplacian")
	fun.add_operator(op)

	bc = sfem.DirichletConditions(fs)
	sfem.add_condition(bc, sinlet, 0, -1);
	# sfem.add_condition(bc, soutlet, 0, 1);
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
	sfem.gradient(fun, x, g)
	c = np.zeros(fs.n_dofs())
	sfem.apply(cg, g, c)
	x -= c

	sfem.report_solution(fun, x)

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

	solve_poisson(options)
