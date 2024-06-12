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
	p0 = vec3(0, 0, 0)
	p1 = vec3(1, 0, 0)
	p2 = vec3(0, 1, 0)
	p3 = vec3(0, 0, 1)

	p4 = (p0 + p1)/2
	p5 = (p1 + p2)/2
	p6 = (p0 + p2)/2
	p7 = (p0 + p3)/2
	p8 = (p1 + p3)/2
	p9 = (p2 + p3)/2

	# Tet4
	v = v.subs(x0, p0[0])
	v = v.subs(y0, p0[1])
	v = v.subs(z0, p0[2])

	v = v.subs(x1, p1[0])
	v = v.subs(y1, p1[1])
	v = v.subs(z1, p1[2])

	v = v.subs(x2, p2[0])
	v = v.subs(y2, p2[1])
	v = v.subs(z2, p2[2])

	v = v.subs(x3, p3[0])
	v = v.subs(y3, p3[1])
	v = v.subs(z3, p3[2])

	# Tet10
	v = v.subs(x4, p4[0])
	v = v.subs(y4, p4[1])
	v = v.subs(z4, p4[2])

	v = v.subs(x5, p5[0])
	v = v.subs(y5, p5[1])
	v = v.subs(z5, p5[2])

	v = v.subs(x6, p6[0])
	v = v.subs(y6, p6[1])
	v = v.subs(z6, p6[2])

	v = v.subs(x7, p7[0])
	v = v.subs(y7, p7[1])
	v = v.subs(z7, p7[2])

	v = v.subs(x8, p8[0])
	v = v.subs(y8, p8[1])
	v = v.subs(z8, p8[2])

	v = v.subs(x9, p9[0])
	v = v.subs(y9, p9[1])
	v = v.subs(z9, p9[2])

	return sp.simplify(v)
	# return v.subs(qx,0).subs(qy,0).subs(qz,0)


def matrix_elem_sub(mat):
	rows, cols = mat.shape
	ret = sp.zeros(rows, cols)

	for i in range(0, rows):
		for j in range(0, cols):
			ret[i, j] = value_elem_sub(mat[i, j])
	return ret


def test_jac():

	p0 = vec3(0, 0, 0)
	p1 = vec3(1, 0, 0)
	p2 = vec3(0, 1, 0)
	p3 = vec3(0, 0, 1)

	p4 = (p0 + p1)/2
	p5 = (p1 + p2)/2
	p6 = (p0 + p2)/2
	p7 = (p0 + p3)/2
	p8 = (p1 + p3)/2
	p9 = (p2 + p3)/2

	p = [p0, p1, p2, p3, 
		 p4, p5, p6, p7, p8, p9]

	fe = Tet10()
	q = vec3(qx, qy, qz)
	g = fe.grad(q)

	ret = sp.zeros(3, 3)

	for i in range(0, 10):
		for d1 in range(0, 3):
			for d2 in range(0, 3):
				ret[d1, d2] += p[i][d1] * g[i][d2]

	print(determinant(ret))

	cc = fe.coords()
	symb = sp.zeros(3, 3)
	for i in range(0, 10):
		for d1 in range(0, 3):
			for d2 in range(0, 3):
				symb[d1, d2] += cc[d1][i] * g[i][d2]

	actual = matrix_elem_sub(fe.isoparametric_jacobian(q))
	symb = matrix_elem_sub(symb)
	print(determinant(symb))
	print(determinant(actual))

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

	def l2_projection_utils(self):
		q = vec3(qx, qy, qz)
		test_fun = self.fun(q)

		J = self.fe.isoparametric_jacobian(q)
		det_J = determinant(J)

		J_numeric = matrix_elem_sub(J)
		print(f'Check J: {determinant(J_numeric)}')


		print("------ measure -------")
		expr = [ast.Assignment(sp.symbols('measure'), det_J*self.fe.reference_measure())]
		c_code(expr)

		print("------ isoparametric_transform -------")
		c_code(assign_matrix("p", self.fe.isoparametric_transform(q)))


		print("------ qp diag -------")

		fun = self.fe.fun(q)
		dV = sp.symbols('dV')
		d = sp.zeros(fe.n_nodes(), 1)

		for i in range(0, fe.n_nodes()):
			d[i] = sp.simplify(fun[i] * test_fun[i] * dV)

		c_code(assign_matrix("element_diag", d))
			

	def check(self):
		q = vec3(qx, qy, qz)
		test_fun = self.fun(q)
		fun 	 = self.fe.fun(q)

		dV = fe.jacobian_determinant(q)
		M = sp.zeros(fe.n_nodes(), fe.n_nodes())

		J = fe.isoparametric_jacobian(q)
		J = matrix_elem_sub(J)

		print_matrix("J", J)

		det_J = determinant(J)
		det_J = sp.simplify(det_J)

		print(f'det_J = {det_J.subs(qx,0).subs(qy,0).subs(qz,0)}')

		for i in range(0, fe.n_nodes()):
			for j in range(0, fe.n_nodes()):
				M[i, j] = fe.integrate(q, fun[i] * test_fun[j]) * dV

		sum_diag = 0		
		for i in range(0, fe.n_nodes()):
			sum_diag += M[i, i]

		print("sum_diag: ", value_elem_sub(sum_diag))
		print_matrix("lumped", matrix_elem_sub(M));

if __name__ == '__main__':
	test_jac()
	fe = DSFE(Tet10())
	# # fe = DSFE(Tri6())
	# fe.check()

	fe.l2_projection_utils()
	# fe.generate_mass_vector()
	# fe.generate_qp_based_code()

