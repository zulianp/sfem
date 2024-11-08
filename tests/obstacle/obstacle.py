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
use_penalty = False

def rigid_body_modes(m):
	x = sfem.points(m, 1)
	y = sfem.points(m, 2)
	z = sfem.points(m, 2)

	n = m.n_nodes()

	e0 = np.zeros(n * 3)
	e1 = np.zeros(n * 3)
	e2 = np.zeros(n * 3)
	e3 = np.zeros(n * 3)
	e4 = np.zeros(n * 3)
	e5 = np.zeros(n * 3)

	# Translation
	e0[range(0, n*3, 3)] = 1
	
	e1[range(1, n*3, 3)] = 1
	
	e2[range(2, n*3, 3)] = 1

	# Rotation
	e3[range(1, n*3, 3)] = -z
	e3[range(2, n*3, 3)] = y
	
	e4[range(0, n*3, 3)] = z
	e4[range(2, n*3, 3)] = -x

	e5[range(0, n*3, 3)] = -y
	e5[range(1, n*3, 3)] = x

	return e0, e1, e2, e3, e4, e5 
	
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
	sfem.hessian_crs(fun, sfem.view(x), sfem.view(rowptr), sfem.view(colidx),  sfem.view(values))
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

def assemble_crs_and_write(fun, x, path):
	fs = fun.space()
	crs_graph = fun.crs_graph()
	rowptr = crs_graph.rowptr()
	colidx = crs_graph.colidx()
	values = sfem.create_real_buffer(crs_graph.nnz())
	x_buff = sfem.view(x)
	sfem.hessian_crs(fun, x_buff, rowptr, colidx, values)

	# print(sfem.len(colidx), " ", sfem.numpy_view(colidx).ctypes.data)

	if not os.path.exists(path):
		os.mkdir(f'{path}')
		
	sfem.numpy_view(rowptr).tofile(f'{path}/rowptr.raw')
	sfem.numpy_view(colidx).tofile(f'{path}/colidx.raw')
	sfem.numpy_view(values).tofile(f'{path}/values.raw')

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
	print('#ndofs ', fs.n_dofs())
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
		# dps = d + s
		dps = s

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

def solve_obstacle(problem):
	path = problem.input_mesh

	if not os.path.exists(problem.output_dir):
		os.mkdir(f'{problem.output_dir}')

	n = 4
	h = 1./(n - 1)
	wall = 0.6 # The obstacle wall is a x = 0.6

	fun = problem.setup()
	m = fun.space().mesh()
	sobstacle = problem.sobstacle
	contact_surf = problem.contact_surf

	fs = fun.space()
	dim = m.spatial_dimension()
	out = fun.output()

	x = np.zeros(fs.n_dofs(), dtype=real_t)

	# --------------------------------------
	# Obstacle
	# --------------------------------------
	obs = np.ones(fs.n_dofs(), dtype=real_t) * 10000
	constrained_dofs = sobstacle[:] * fs.block_size()

	sy = sfem.points(m, 1)

	if m.spatial_dimension() > 2:
		sz = sfem.points(m, 2)
	else:
		sz = sy * 0

	sdf = problem.sdf(sy, sz)
	obs[constrained_dofs] = sdf[sobstacle]

	for d in range(1, fs.block_size()):
		obs[d::fs.block_size()] = 10000

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


class Cylinder:
	def __init__(self):
		self.input_mesh = './Cylinder'
		self.output_dir = './output'
		self.op = "LinearElasticity"

	def sdf(self, sy, sz):
		radius = ((0.5 - np.sqrt(sy*sy + sz*sz)))
		f = -0.1*np.cos(np.pi*2*radius) - 0.1
		f += -0.05*np.cos(np.pi*8*radius)
		f += -0.01*np.cos(np.pi*16*radius)
		parabola = -indentation * f + wall

		sdf = (parabola - sfem.points(m, 0)).astype(real_t)
		return sdf


	def setup(self):
		path = self.input_mesh
		
		m = sfem.Mesh()		
		m.read(path)

		if self.op == "LinearElasticity":
			print("Setting up LinearElasticity ...")
			dim = m.spatial_dimension()
			fs = sfem.FunctionSpace(m, dim)
			fun = sfem.Function(fs)
			elasticity = sfem.create_op(fs, "LinearElasticity")
			fun.add_operator(elasticity)

			bc = sfem.DirichletConditions(fs)

			sdirichlet = np.unique(np.fromfile(f'{path}/sidesets_aos/sinlet.raw', dtype=idx_t))
			sfem.add_condition(bc, sdirichlet, 0, 0.2);
			sfem.add_condition(bc, sdirichlet, 1, 0.0);

			if dim > 2:
				sfem.add_condition(bc, sdirichlet, 2, 0.);

			fun.add_dirichlet_conditions(bc)

			self.sobstacle = np.fromfile(f'{path}/sidesets_aos/soutlet.raw', dtype=idx_t)
			self.contact_surf = sfem.mesh_connectivity_from_file(f'{path}/surface/outlet')
		else:
			fs = sfem.FunctionSpace(m, 1)
			laplacian = sfem.create_op(fs, "Laplacian")

			fun = sfem.Function(fs)
			fun.add_operator(laplacian)

			bc = sfem.DirichletConditions(fs)

			sdirichlet = np.unique(np.fromfile(f'{path}/sidesets_aos/sinlet.raw', dtype=idx_t))
			sfem.add_condition(bc, sdirichlet, 0, 1);
			fun.add_dirichlet_conditions(bc)

			self.sobstacle = np.fromfile(f'{path}/sidesets_aos/soutlet.raw', dtype=idx_t)
			self.contact_surf = sfem.mesh_connectivity_from_file(f'{path}/surface/outlet')

		print(f"Mesh #nodes {m.n_nodes()}, #elements {m.n_elements()}")
		print(f"FunctionSpace #dofs {fs.n_dofs()}")
		fun.set_output_dir(self.output_dir)
		return fun


class Box3D:
	def __init__(self):
		self.input_mesh = './Box3D'
		self.output_dir = './output'
		self.op = "LinearElasticity"

	def sdf(self, sy, sz):
		sdf = sz * 0 + 0.1
		return sdf
	def setup(self):
		path = self.input_mesh
		
		m = sfem.Mesh()		
		m.read(path)

		if self.op == "LinearElasticity":
			print("Setting up LinearElasticity ...")
			dim = m.spatial_dimension()
			fs = sfem.FunctionSpace(m, dim)
			fun = sfem.Function(fs)
			elasticity = sfem.create_op(fs, "LinearElasticity")
			fun.add_operator(elasticity)

			bc = sfem.DirichletConditions(fs)

			sdirichlet = np.unique(np.fromfile(f'{path}/boundary_nodes/left.int32.raw', dtype=idx_t))
			sfem.add_condition(bc, sdirichlet, 0, 0.2);
			sfem.add_condition(bc, sdirichlet, 1, 0.0);

			if dim > 2:
				sfem.add_condition(bc, sdirichlet, 2, 0.);

			fun.add_dirichlet_conditions(bc)

			self.sobstacle = np.fromfile(f'{path}/boundary_nodes/right.int32.raw', dtype=idx_t)
			self.contact_surf = sfem.mesh_connectivity_from_file(f'{path}/surface/right')
		
		print(f"Mesh #nodes {m.n_nodes()}, #elements {m.n_elements()}")
		print(f"FunctionSpace #dofs {fs.n_dofs()}")
		fun.set_output_dir(self.output_dir)
		return fun

if __name__ == '__main__':
	print(sys.argv)
	if len(sys.argv) < 3:
		print(f'usage: {sys.argv[0]} <case>')
		exit(1)

	sfem.init()

	problems = {
		"Cylinder" : Cylinder(),
		"Box3D" : Box3D()
	}

	problem = problems[sys.argv[1]]
	
	try:
	    opts, args = getopt.getopt(
	        sys.argv[3:], "ho:",
	        ["help","output="])

	except getopt.GetoptError as err:
	    print(err)
	    print(usage)
	    sys.exit(1)

	for opt, arg in opts:
	    if opt in ('-h', '--help'):
	        print(usage)
	        sys.exit()
	    elif opt in ('-p', '--problem'):
	     	problem.op  = arg
	    elif opt in ('-o', '--output'):
	      	problem.output = arg

	solve_obstacle(problem)
	sfem.finalize()
