#!/usr/bin/env python3

# SFEM imports 
import pysfem as sfem
from sfem.sfem_config import *

import numpy as np
from numpy import linalg
import sys, getopt, os

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
	function = sfem.Function(space)
	function.add_operator(sfem.create_op(space, name_op))

	x = np.zeros(space.n_dofs(), dtype=real_t)
	A = assemble_scipy_matrix(function, x)

	print(A)


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
