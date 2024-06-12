#!/usr/bin/env python3

from fe import FE
from sfem_codegen import *
from weighted_fe import *

from tri6 import *
from tet10 import *
from fields import *
# from mass_op import MassOp

def print_matrix(name, mat):
	rows, cols = mat.shape
	expr = []
	print(f'{name}:')
	for i in range(0, rows):
		for j in range(0, cols):
			print(f'{mat[i, j]}', end="\t")
		print("")

def assign_matrix(name, mat):
	rows, cols = mat.shape
	expr = []
	for i in range(0, rows):
		for j in range(0, cols):
			var = sp.symbols(f'{name}[{i*cols + j}]')
			expr.append(ast.Assignment(var, mat[i, j]))
	return expr

def assign_matrix_diag(name, mat):
	rows, cols = mat.shape
	expr = []
	for i in range(0, rows):
			var = sp.symbols(f'{name}[{i}]')
			expr.append(ast.Assignment(var, mat[i, i]))
	return expr

def value_elem_sub(v):
	v = v.subs(x0, 0)
	v = v.subs(y0, 0)
	v = v.subs(z0, 0)

	v = v.subs(x1, 1)
	v = v.subs(y1, 0)
	v = v.subs(z1, 0)

	v = v.subs(x2, 0)
	v = v.subs(y2, 1)
	v = v.subs(z2, 0)

	v = v.subs(x3, 0)
	v = v.subs(y3, 0)
	v = v.subs(z3, 1)
	return v

def matrix_elem_sub(mat):
	rows, cols = mat.shape
	ret = sp.zeros(rows, cols)

	for i in range(0, rows):
		for j in range(0, cols):
			ret[i, j] = value_elem_sub(mat[i, j])

	return ret



class DSFE(WeightedFE):
	def __init__(self, fe, prefix_ = "DSFE"):
		self.fe = fe
		weights = sp.zeros(fe.n_nodes(), fe.n_nodes())

		for i in range(0, fe.n_nodes()):
			weights[i, i] = 1


		M = sp.zeros(fe.n_nodes(), fe.n_nodes())
		
		q = [qx, qy, qz]
		fun = fe.fun(q)
		dV = fe.jacobian_determinant(q)

		expr = []
		sum_mat = 0
		
		for i in range(0, fe.n_nodes()):
			for j in range(0, fe.n_nodes()):
				M[i, j] = fe.integrate(q, fun[i] * fun[j]) * dV
				sum_mat += M[i, j]

		sum_diag = 0
		D = sp.zeros(fe.n_nodes(), fe.n_nodes())

		for i in range(0, fe.n_nodes()):
			D[i, i] = M[i, i]
			sum_diag += D[i, i]

		# Analytic result is numerical
		scaling_factor = sp.simplify(sum_mat/sum_diag)
		print("sum_diag: ", value_elem_sub(sum_diag),", ", sp.simplify(sum_diag))
		print("sum_mat: ", value_elem_sub(sum_mat), ", ", sp.simplify(sum_mat), )
		print("scaling_factor: ", scaling_factor)

		# Diagonal scaling 
		for i in range(0, fe.n_nodes()):
			D[i, i] *= scaling_factor

		# print_matrix("M", matrix_elem_sub(M));
		# print_matrix("D", matrix_elem_sub(D));

		M_numeric = matrix_elem_sub(M)
		D_numeric = matrix_elem_sub(D)
		M_numeric_inv = M_numeric.inv()

		W = (M_numeric_inv * D_numeric).T

		self.M = M
		self.D = D

		# print_matrix("W", W)
		super().__init__(fe, W, prefix_)

	def generate_mass_vector(self):
		D = self.D
		expr = assign_matrix_diag("diag", D)
		print("------ lumped mass vector -------")
		c_code(expr)

	def check(self):
		q = [qx, qy, qz]
		test_fun = self.fun(q)
		fun 	 = self.fe.fun(q)

		dV = fe.jacobian_determinant(q)

		M = sp.zeros(fe.n_nodes(), fe.n_nodes())

		for i in range(0, fe.n_nodes()):
			for j in range(0, fe.n_nodes()):
				M[i, j] = fe.integrate(q, fun[i] * test_fun[j]) * dV

		sum_diag = 0		
		for i in range(0, fe.n_nodes()):
			sum_diag += M[i, i]

		print("sum_diag: ", value_elem_sub(sum_diag))
		print_matrix("lumped", matrix_elem_sub(M));

if __name__ == '__main__':
	fe = DSFE(Tet10())
	fe.generate_mass_vector()
	# fe = DSFE(Tri6())
	# fe.check()
	fe.generate_qp_based_code()

