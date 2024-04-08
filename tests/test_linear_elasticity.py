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
	m.convert_to_macro_element_mesh()

	sinlet = np.fromfile(f'{path}/sidesets_aos/sinlet.raw', dtype=idx_t)
	soutlet = np.fromfile(f'{path}/sidesets_aos/soutlet.raw', dtype=idx_t)

	fs = s.FunctionSpace(m, m.spatial_dimension())
	fun = s.Function(fs)
	fun.set_output_dir(options.output_dir)

	op = s.create_op(fs, "LinearElasticity")
	fun.add_operator(op)

	bc = s.DirichletConditions(fs)
	s.add_condition(bc, sinlet, 0, 0.1);
	s.add_condition(bc, soutlet, 0, -0.1);
	fun.add_dirichlet_conditions(bc)

	# nc = s.NeumannConditions(fs)
	# s.add_condition(nc, soutlet, 0, 1)
	# fun.add_operator(nc)

	x = np.zeros(fs.n_dofs())
	g = np.zeros(fs.n_dofs())

	s.apply_constraints(fun, x)

	tol = 1e-8
	alpha = 0.01
	max_it = 100000
	for k in range(0, max_it):
		g.fill(0)
		s.gradient(fun, x, g)
		s.apply_zero_constraints(fun, g)

		x -= alpha * g

		norm_g = linalg.norm(g)
		stop = norm_g < tol or k == max_it - 1 
		if np.mod(k, 100000) == 0 or stop:
			# val = s.value(fun, x)
			print(f'{k}) norm(g) = {norm_g}')
			s.report_solution(fun, x)

		if stop:
			break

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

	solve_poisson(options)
