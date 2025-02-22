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

def assign_tensor3(name, mat):
	rows, cols = mat.shape
	expr = []
	for zi in range(0, 3):
		for yi in range(0, 3):
			for xi in range(0, 3):
				var = sp.symbols(f'{name}[{xi}][{yi}][{zi}]')
				val = mat[zi * 9 + yi * 3 + xi]
				expr.append(ast.Assignment(var, val))
	return expr

if __name__ == '__main__':
	fe = Quad4(True)
	# fe = Hex8(False)
	dim = fe.spatial_dim()

	op = LaplaceOp(fe, True)
	f = Field(fe, coeffs('u', fe.n_nodes()))
	# op = MassOp(f, fe, True)

	# L = op.sym_matrix()
	
	# L = matrix_coeff('A', fe.n_nodes(), fe.n_nodes())
	L = sym_matrix_coeff('A', fe.n_nodes(), fe.n_nodes())
	S = fe.to_stencil(L)
 	# S = fe.to_masked_stencil(L)
	# print(S.shape)
	
	# Su = S * coeffs('u', S.shape[1])
	# print('//DIFF = ', matrix_sum(S) - matrix_sum(L))
	# expr = []
	for k,v in S.items():
		print(f'===============\n{k})\n===============')
		expr = assign_matrix(k, v)
		# expr = assign_tensor3("S", S)
		# expr = assign_matrix("Su", Su)
		c_code(expr)

	# c_code(op.hessian())
