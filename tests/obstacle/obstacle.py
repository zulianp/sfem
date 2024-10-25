#!/usr/bin/env python3

import pysfem as sfem
import numpy as np
from numpy import linalg
import sys, getopt, os

from sfem.sfem_config import *

# --------------------------------------
# Solver parameters
# --------------------------------------
MAX_NL_ITER = 1000
max_linear_iterations = 8
penalty_param = 1 # Very sensitive to this! If few linear iterations it is less sensitive
use_cheb = False
matrix_free = True
use_penalty = True

def create_mg():
	mg = sfem.Multigrid()
	# Example 2-level
	# mg.add_level(linear_op, smoother, None, restriction);
	# mg.add_level(None, solver_coarse, prolongation, None);
	return mg

# Assemble matrix in scipy format
def assemble_scipy_matrix(fun, x):
	import scipy

	fs = fun.space()
	crs_graph = fun.crs_graph()
	rowptr = sfem.numpy_view(crs_graph.rowptr())
	colidx = sfem.numpy_view(crs_graph.colidx())
	values = np.zeros(colidx.shape, dtype=real_t)
	sfem.hessian_crs(fun,  sfem.view(x), sfem.view(rowptr),  sfem.view(colidx),  sfem.view(values))
	M = scipy.sparse.csr_matrix((values, colidx, rowptr), shape=(fs.n_dofs(), fs.n_dofs())) 
	return M

def assemble_crs_spmv(fun, x):
	fs = fun.space()
	crs_graph = fun.crs_graph()
	rowptr = crs_graph.rowptr()
	colidx = crs_graph.colidx()
	values = sfem.create_real_buffer(crs_graph.nnz())
	x_buff = sfem.view(x)
	sfem.hessian_crs(fun, x_buff, rowptr, colidx, values)
	return sfem.crs_spmv(rowptr, colidx, values)

def solve_shifted_penalty(fun, contact_surf, constrained_dofs, obs, x, out):
	fs = fun.space()

	# --------------------------------------
	# Selector for constrained dofs
	# --------------------------------------
	selector = np.zeros(fs.n_dofs())
	selector[constrained_dofs] = 1

	# --------------------------------------
	# Boundary Mass Matrix (with lumping)
	# --------------------------------------
	mass_op = sfem.create_boundary_op(fs, contact_surf, "BoundaryMass")
	mass = np.zeros(fs.n_dofs(), dtype=real_t)
	ones = np.ones(fs.n_dofs(), dtype=real_t)
	boundary_fun = sfem.Function(fs)
	boundary_fun.add_operator(mass_op)
	boundary_op = sfem.make_op(boundary_fun, x)
	sfem.apply(boundary_op, ones, mass)

	# --------------------------------------
	# Linear solver
	# --------------------------------------
	fun_diag = np.zeros(fs.n_dofs(), dtype=real_t)
	sfem.hessian_diag(fun, x, fun_diag)

	if matrix_free:
		# Matrix-free
		lop = sfem.make_op(fun, x)
	else:
		# Matrix-based
		lop = assemble_crs_spmv(fun, x)

	print(assemble_scipy_matrix(fun, x))

	if use_cheb:
		solver = sfem.Chebyshev3()
		inv_diag = sfem.diag(1./np.sqrt(fun_diag))
		solver.set_op(inv_diag * lop * inv_diag)
		solver.default_init()
		solver.init_with_ones()
		solver.set_max_it(max_linear_iterations)
	else:
		solver = sfem.ConjugateGradient()
		solver.default_init()
		solver.set_max_it(max_linear_iterations)
		solver.set_rtol(1e-2)
		solver.set_atol(1e-12)
		solver.set_verbose(False)

	# --------------------------------------
	# Allocate buffers
	# --------------------------------------
	g = np.zeros(fs.n_dofs(), dtype=real_t)
	g_pen = np.zeros(fs.n_dofs(), dtype=real_t)

	c = np.zeros(fs.n_dofs(), dtype=real_t)

	sfem.apply_constraints(fun, x)
	s = np.zeros(fs.n_dofs(), dtype=real_t)
	r = np.zeros(fs.n_dofs(), dtype=real_t)

	# --------------------------------------
	# Nonlinear iteration
	# --------------------------------------
	for i in range(0, MAX_NL_ITER):
		# Material contribution
		g[:] = 0.
		sfem.gradient(fun, x, g)
		
		# Shifted-Penalty contribution
		g_pen[:] = 0.
		d = (obs - x)
		dps = d + s
		active = selector * (dps <= 0)
		s = active * dps
		dps = d + s

		H_diag = (penalty_param * mass * active)
		sfem.apply_zero_constraints(fun, H_diag)
		H = sfem.diag(H_diag)
		op = H + lop

		g_pen = (penalty_param * mass * active * dps)
		sfem.apply_zero_constraints(fun, g_pen)
		
		if use_cheb:
			# Cheb
			diag_12 = np.sqrt(fun_diag + H_diag)
			inv_diag = sfem.diag(1./diag_12)

			r[:] = 0.
			sfem.apply(inv_diag, (g_pen - g), r)
			solver.set_op(inv_diag * op * inv_diag)
		else:
			# CG
			inv_diag = sfem.diag(1./(fun_diag + H_diag))
			solver.set_preconditioner_op(inv_diag)
			r = g_pen - g
			solver.set_op(op)

		c[:] = 0.
		sfem.copy_constrained_dofs(fun, r, c)
		sfem.apply(solver, r, c)

		if use_cheb:
			sfem.apply(inv_diag, c, x)
		else:
			x += c

		norm_g = linalg.norm(g_pen - g)
		norm_penet = linalg.norm(active*d)
		print(f'{i}) norm_g = {norm_g} #active {int(np.sum(active))} norm_penet = {norm_penet}')

		if(norm_g < 1e-8):
			break

	sfem.gradient(fun, x, g)
	sfem.write(out, "g", g)
	sfem.write(out, "H_diag", H_diag)
	sfem.write(out, "violation", d * active)
	sfem.write(out, "selector", selector)
	return x

def solve_obstacle(options):
	path = options.input_mesh

	if not os.path.exists(options.output_dir):
		os.mkdir(f'{options.output_dir}')

	n = 4
	h = 1./(n - 1)
	wall = 0.6 # The obstacle wall is a x = 0.6

	m = sfem.Mesh()		
	m.read(path)
	sdirichlet = np.unique(np.fromfile(f'{path}/sidesets_aos/sinlet.raw', dtype=idx_t))
	sobstacle = np.fromfile(f'{path}/sidesets_aos/soutlet.raw', dtype=idx_t)

	contact_surf = sfem.mesh_connectivity_from_file(f'{path}/surface/outlet')

	dim = m.spatial_dimension()
	fs = sfem.FunctionSpace(m, dim)
	fun = sfem.Function(fs)
	
	fun.set_output_dir(options.output_dir)
	out = fun.output()

	elasticity = sfem.create_op(fs, "LinearElasticity")
	fun.add_operator(elasticity)

	bc = sfem.DirichletConditions(fs)
	sfem.add_condition(bc, sdirichlet, 0, 0.2);
	sfem.add_condition(bc, sdirichlet, 1, 0.0);

	if dim > 2:
		sfem.add_condition(bc, sdirichlet, 2, 0.);

	fun.add_dirichlet_conditions(bc)
	x = np.zeros(fs.n_dofs(), dtype=real_t)

	# --------------------------------------
	# Obstacle
	# --------------------------------------
	obs = np.ones(fs.n_dofs(), dtype=real_t) * 10000
	constrained_dofs = sobstacle[:] * dim

	sy = sfem.points(m, 1)
	sz = sfem.points(m, 2)

	indentation = 1

	radius = ((0.5 - np.sqrt(sy*sy + sz*sz)))
	f = -0.1*np.cos(np.pi*2*radius) - 0.1
	# f += -0.05*np.cos(np.pi*8*radius)
	parabola = -indentation * f + wall

	sdf = (parabola - sfem.points(m, 0)).astype(real_t)
	obs[constrained_dofs] = sdf[sobstacle]

	for d in range(1, dim):
		obs[d::dim] = 10000

	# --------------------------------------
	# Solve obstacle problem
	# --------------------------------------
	if use_penalty:
		solve_shifted_penalty(fun, contact_surf, constrained_dofs, obs, x, out)
	else:
		solver = sfem.MPRGP()
		solver.default_init()
		solver.set_atol(1e-12)
		solver.set_rtol(1e-6);
		solver.set_max_it(2000)

		if matrix_free:
			# Matrix-free
			lop = sfem.make_op(fun, x)
		else:
			# Matrix-based
			lop = assemble_crs_spmv(fun, x)

		solver.set_op(lop)

		c = np.zeros(fs.n_dofs(), dtype=real_t)
		g = np.zeros(fs.n_dofs(), dtype=real_t)

		for i in range(0, MAX_NL_ITER):
			obs_diff = obs - x
			sfem.set_upper_bound(solver, obs_diff)

			g[:] = 0.
			sfem.gradient(fun, x, g)
			
			c[:] = 0.
			sfem.apply(solver, -g, c)

			x += c

			norm_c = linalg.norm(c)

			print(f'{i}) norm_c {norm_c}')
			if norm_c < 1e-8:
				break
	
	# --------------------------------------
	# Write output
	# --------------------------------------
	sfem.write(out, "obs", obs)
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
