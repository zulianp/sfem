#!/usr/bin/env python3

# SFEM imports 
import pysfem as sfem

import numpy as np
from numpy import linalg
import sys, getopt, os

# Device not yet supported
execution_space = sfem.ExecutionSpace.EXECUTION_SPACE_HOST

def create_default_mesh_and_operator():
	"""
	Create a default hex8 cube mesh when no arguments are provided.
	"""
	print("No arguments provided. Creating default hex8 cube mesh...")
	# Create a simple hex8 cube mesh (2x2x2 elements)
	mesh = sfem.create_hex8_cube(2, 2, 2, 0, 0, 0, 1, 1, 1)
	# Use Laplacian operator for the default case
	name_op = "Laplacian"
	print(f"Created default mesh: {mesh.n_nodes()} nodes, {mesh.n_elements()} elements")
	print(f"Using operator: {name_op}")
	return mesh, name_op

def assemble_scipy_matrix(fun, x):
	import scipy
	space = fun.space()
	crs_graph = fun.crs_graph()
	rowptr = sfem.numpy_view(crs_graph.rowptr())
	colidx = sfem.numpy_view(crs_graph.colidx())
	values = np.zeros(colidx.shape, dtype=sfem.real_t)
	sfem.hessian_crs(fun, sfem.view(x), sfem.view(rowptr), sfem.view(colidx),  sfem.view(values))
	M = scipy.sparse.csr_matrix((values, colidx, rowptr), shape=(space.n_dofs(), space.n_dofs())) 
	return M

def run_amg_test(mesh, name_op):
	dim = mesh.spatial_dimension()
	block_size = 1
	if name_op == "LinearElasticity":
		block_size = dim
	space = sfem.FunctionSpace(mesh, block_size)
	print(f'COARSE: #dofs {space.n_dofs()}')
	space.promote_to_semi_structured(8)
	print(f'FINE: #dofs {space.n_dofs()}')
	coarse_space = space.derefine(1)
	
	prolongation = sfem.create_hierarchical_prolongation(coarse_space, space, execution_space)
	restriction =  sfem.create_hierarchical_restriction(space, coarse_space, execution_space)
	
	function = sfem.Function(space)
	function.add_operator(sfem.create_op(space, name_op, execution_space))
	
	coarse_function = sfem.Function(coarse_space)
	coarse_function.add_operator(sfem.create_op(coarse_space, name_op, execution_space))
	
	x = np.zeros(space.n_dofs(), dtype=sfem.real_t)
	r = np.zeros(space.n_dofs(), dtype=sfem.real_t)
	rng = np.random.default_rng()
	coarse_x = rng.standard_normal(coarse_space.n_dofs(), dtype=sfem.real_t)
	coarse_x /= np.max(coarse_x)
	coarse_r = np.zeros(coarse_space.n_dofs(), dtype=sfem.real_t)
	coarse_r_galerkin = np.zeros(coarse_space.n_dofs(), dtype=sfem.real_t)

	linear_op = sfem.make_op(function, sfem.view(x))
	coarse_linear_op = sfem.make_op(coarse_function, sfem.view(coarse_x))

	sfem.apply(coarse_linear_op, sfem.view(coarse_x), sfem.view(coarse_r))
	sfem.apply(prolongation, sfem.view(coarse_x), sfem.view(x))
	sfem.apply(linear_op, sfem.view(x), sfem.view(r))
	sfem.apply(restriction, sfem.view(r), sfem.view(coarse_r_galerkin))
	print(f'Diff = {np.sum(np.abs(coarse_r - coarse_r_galerkin))}')
	# TODO: coarse matrix, CRS_SYM, Smoother, MG, etc.

if __name__ == '__main__':
	sfem.init()
	print(sys.argv)
	if len(sys.argv) < 3:
		print(f'usage: {sys.argv[0]} <mesh> <op=Laplacian|LinearElasticity>')
		print("No arguments provided. Using default hex8 cube mesh with Laplacian operator.")
		mesh, name_op = create_default_mesh_and_operator()
		run_amg_test(mesh, name_op)
	else:
		path_mesh = sys.argv[1]
		name_op = sys.argv[2]
		mesh = sfem.Mesh()
		mesh.read(path_mesh)
		run_amg_test(mesh, name_op)
	sfem.finalize()
