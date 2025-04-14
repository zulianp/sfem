#!/usr/bin/env python3

# SFEM imports 
import pysfem as sfem
from sfem.sfem_config import *

import numpy as np
from numpy import linalg
import sys, getopt, os

# Device not yet supported
execution_space = sfem.ExecutionSpace.EXECUTION_SPACE_HOST

def assemble_scipy_matrix(fun, x):
	import scipy

	space = fun.space()
	crs_graph = fun.crs_graph()
	rowptr = sfem.numpy_view(crs_graph.rowptr())
	colidx = sfem.numpy_view(crs_graph.colidx())
	values = np.zeros(colidx.shape, dtype=real_t)
	sfem.hessian_crs(fun, sfem.view(x), sfem.view(rowptr), sfem.view(colidx),  sfem.view(values))
	M = scipy.sparse.csr_matrix((values, colidx, rowptr), shape=(space.n_dofs(), space.n_dofs())) 
	return M

def run(path_mesh, name_op):
	mesh = sfem.Mesh()		
	mesh.read(path_mesh)

	dim = mesh.spatial_dimension()
	
	block_size = 1
	if name_op == "LinearElasticity":
		block_size = dim

	space = sfem.FunctionSpace(mesh, block_size)
	print(f'COARSE: #dofs {space.n_dofs()}')
	# Semi-structured discretization enabled here (param=levels)
	space.promote_to_semi_structured(8) 
	print(f'FINE: #dofs {space.n_dofs()}')

	coarse_space = space.derefine(1)

	# Geometric P/R
	prolongation = sfem.create_hierarchical_prolongation(coarse_space, space, execution_space)
	restriction =  sfem.create_hierarchical_restriction(space, coarse_space, execution_space)

	# Fine operator
	function = sfem.Function(space)
	function.add_operator(sfem.create_op(space, name_op))

	# Coarse operator
	coarse_function = sfem.Function(coarse_space)
	coarse_function.add_operator(sfem.create_op(coarse_space, name_op))

	x = np.zeros(space.n_dofs(), dtype=real_t)
	r = np.zeros(space.n_dofs(), dtype=real_t)
	
	rng = np.random.default_rng()
	coarse_x = rng.standard_normal(coarse_space.n_dofs(), dtype=real_t)
	coarse_x /= np.max(coarse_x)
	coarse_r = np.zeros(coarse_space.n_dofs(), dtype=real_t)
	coarse_r_galerkin = np.zeros(coarse_space.n_dofs(), dtype=real_t)

	linear_op = sfem.make_op(function, x)
	coarse_linear_op = sfem.make_op(coarse_function, coarse_x)

	# Coarse operator
	sfem.apply(coarse_linear_op, coarse_x, coarse_r)

	# Galerkin products
	sfem.apply(prolongation, coarse_x, x)
	sfem.apply(linear_op, x, r)
	sfem.apply(restriction, r, coarse_r_galerkin)
	
	print(f'Diff = {np.sum(np.abs(coarse_r - coarse_r_galerkin))}')

	# TODO 
	# - coarse matrix (crs|coo)
	# - CRS_SYM D, U; y = C(x, u) * x + B(x) * (D*x + U*x + U^T*x) (Neumann operator)
	# - Smoother
	# - MG
	# A = assemble_scipy_matrix(coarse_function, coarse_x)
	# print(A)

if __name__ == '__main__':
	sfem.init()

	print(sys.argv)
	if len(sys.argv) < 3:
		print(f'usage: {sys.argv[0]} <mesh> <op=Laplacian|LinearElasticity>')
		exit(1)

	path_mesh = sys.argv[1]
	name_op = sys.argv[2]
	run(path_mesh, name_op)

	sfem.finalize()
