#!/usr/bin/env python3

import pysfem as s
import numpy as np
from numpy import linalg

import sys, getopt

idx_t = np.int32

def solve_poisson(options):
	s.init()
	m = s.Mesh()

	path = options.input_mesh
	m.read(path)

	sinlet = np.fromfile(f'{path}/sidesets_aos/sinlet.raw', dtype=idx_t)
	soutlet = np.fromfile(f'{path}/sidesets_aos/soutlet.raw', dtype=idx_t)

	fs = s.FunctionSpace(m, 1)
	fun = s.Function(fs)
	fun.set_output_dir(options.output_dir)

	op = s.create_op(fs, "Laplacian")
	fun.add_operator(op)

	bc = s.DirichletConditions(fs)
	s.add_condition(bc, sinlet, 0, -1);
	fun.add_dirichlet_conditions(bc)

	nc = s.NeumannConditions(fs)
	s.add_condition(nc, soutlet, 0, 1)
	fun.add_operator(nc)

	x = np.zeros(fs.n_dofs())
	g = np.zeros(fs.n_dofs())

	alpha = 0.1
	max_it = 10000
	for k in range(0, max_it):
		g.fill(0)
		s.gradient(fun, x, g)
		
		x -= alpha * g

		norm_g = linalg.norm(g)
		stop = norm_g < 1e-5
		if np.mod(k, 1000) == 0 or stop:
			val = s.value(fun, x)
			print(f'{k}) v = {val}, norm(g) = {norm_g}')
			s.report_solution(fun, x)

		if stop:
			break

	# cg = s.ConjugateGradient()
	# cg.default_init()
	
	# lop = s.make_op(fun, x)
	# cg.set_op(lop)

	# g.fill(0)
	# s.gradient(fun, x, g)
	# g = -g

	# c = np.zeros(fs.n_dofs())
	# s.apply(cg, g, c)
	# x += c

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
