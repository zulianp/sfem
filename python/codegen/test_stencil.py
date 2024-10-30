#!/usr/bin/env python3

from tri6 import *
from quad4 import *
from mass_op import *
from laplace_op import *


def assign_matrix(name, mat):
	rows, cols = mat.shape
	expr = []
	for i in range(0, rows):
		for j in range(0, cols):
			var = sp.symbols(f'{name}[{i*cols + j}]')
			expr.append(ast.Assignment(var, mat[i, j]))
	return expr

if __name__ == '__main__':
	# fe = Quad4(True)
	fe = Hex8(False)
	dim = fe.spatial_dim()

	# op = LaplaceOp(fe, True)
	f = Field(fe, coeffs('u', fe.n_nodes()))
	op = MassOp(f, fe, True)

	L = op.sym_matrix()
	S = fe.to_masked_stencil(L)

	# print('//DIFF = ', matrix_sum(S) - matrix_sum(L))
 

	expr = assign_matrix("S", S)
	c_code(expr)

	# c_code(op.hessian())
