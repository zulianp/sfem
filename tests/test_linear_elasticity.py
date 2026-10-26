#!/usr/bin/env python3

import pysfem as sfem
import sfem.mesh.box_mesh as box_mesh
import numpy as np
from numpy import linalg
from time import perf_counter

import sys, getopt

idx_t = np.int32

def solve_linear_elasticity(options):
	tick = perf_counter()

	path = options.input_mesh
	if path == "gen:box":
		idx, points = box_mesh.create(2, 1, 1, 20, 10, 10, "tet4")
		m = sfem.create_mesh("TET4", np.array(idx), np.array(points))
	else:
		m = sfem.Mesh.read(path)
		# m.convert_to_macro_element_mesh()

	# Clamped at x = 0, pulled at the other end of the bar.
	x_max = float(np.max(sfem.points(m, 0)))
	eps = 1e-6 * x_max
	sinlet = sfem.Sideset.create_from_selector(m, lambda x, y, z: x < eps)
	soutlet = sfem.Sideset.create_from_selector(m, lambda x, y, z: x > x_max - eps)

	fs = sfem.FunctionSpace(m, m.spatial_dimension())
	fun = sfem.Function(fs)
	fun.set_output_dir(options.output_dir)
	# The displacement is written one component per file.
	fun.output().enable_AoS_to_SoA(True)

	# op = sfem.create_op(fs, "LinearElasticity")
	op = sfem.create_op(fs, "NeoHookeanOgden")
	op.initialize()
	fun.add_operator(op)

	conditions = []
	for component in range(0, m.spatial_dimension()):
		c = sfem.DirichletCondition()
		c.sidesets = sinlet
		c.component = component
		c.value = 0.
		conditions.append(c)
	bc = sfem.create_dirichlet_conditions(fs, conditions, sfem.ExecutionSpace.EXECUTION_SPACE_HOST)
	fun.add_constraint(bc)

	traction = sfem.NeumannCondition()
	traction.sidesets = soutlet
	traction.component = 0
	traction.value = 1
	fun.add_operator(sfem.create_neumann_conditions(fs, [traction], sfem.ExecutionSpace.EXECUTION_SPACE_HOST))

	x = np.zeros(fs.n_dofs())
	# A Buffer over x: the library reads and writes it in place.
	xb = sfem.view(x)

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
		sfem.hessian_diag(fun, xb, sfem.view(d))
		prec = sfem.diag(sfem.view(1./d))
		cg.set_preconditioner_op(prec)

		print(np.max(d))
		print(np.min(d))
		print(np.max(np.abs(d)))
		print(np.min(np.abs(d)))

	# Newton iteration
	g = np.zeros(fs.n_dofs())
	c = np.zeros(fs.n_dofs())
	gb, cb = sfem.view(g), sfem.view(c)
	for i in range(0, 100):
		g[:] = 0
		sfem.apply_constraints(fun, xb)
		sfem.gradient(fun, xb, gb)
		sfem.apply_zero_constraints(fun, gb)

		# As the pointer of x is used not necessary to be in the loop
		# But maybe there will be some side effect in the future
		lop = sfem.make_op(fun, xb)
		cg.set_op(lop)

		c[:] = 0
		sfem.apply(cg, gb, cb)
		x -= damping * c

		norm_g = linalg.norm(g)
		print(f'{i}) norm_g = {norm_g}')
		if(norm_g < 1e-10):
			break

	# Write result to disk
	sfem.report_solution(fun, xb)

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

	sfem.init()
	solve_linear_elasticity(options)
	sfem.finalize()
