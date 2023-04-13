#!/usr/bin/env python3

from sfem_codegen import *

from tet4 import *
from tri3 import *
from tri6 import *
from tri1 import *

from fields import *

class GradAndProjectOp:
	def __init__(self, field, fe_test, q):
		self.q = sp.Matrix(fe_test.manifold_dim(), 1, q)
		self.field = field
		self.fe_test = fe_test

	# def matrix(self):
	# 	field = self.field
	# 	fe_test = self.fe_test
	# 	q = self.q

	# 	J_inv = field.fe.jacobian_inverse(q)
	# 	grad_field = J_inv * field.grad(q)

	# 	f_to   = fe_test.fun(q)

	# 	expr = []

	# 	for i in range(0, fe_test.n_nodes()):
	# 		for j in range(0, field.n_nodes()):
	# 			integr = fe_test.integrate(q, grad_field[j] * f_to[i] * fe_test.jacobian_determinant(q))
	# 			var = sp.symbols(f'element_matrix[{i*fe.n_nodes() + j}*stride]')
	# 			expr.append(ast.Assignment(var, integr))

		# return expr

	def apply(self):
		field = self.field
		fe_test = self.fe_test
		q = self.q

		J_inv = field.fe.jacobian_inverse(q)
		grad_field = J_inv * field.grad(q)
		f_to   = fe_test.fun(q)

		expr = []
		postfix = ['x', 'y', 'z']

		for d in range(0, fe_test.spatial_dim()):
			for i in range(0, fe_test.n_nodes()):
				integr = fe_test.integrate(q, grad_field[d] * f_to[i] * fe_test.jacobian_determinant(q))
				lform = sp.symbols(f'element_vector_{postfix[d]}[{i}*stride]')
				expr.append(ast.Assignment(lform, integr))

		for i in range(0, fe_test.n_nodes()):
			integr = fe_test.integrate(q, f_to[i] * fe_test.jacobian_determinant(q))
			lform = sp.symbols(f'element_vector_w[{i}*stride]')
			expr.append(ast.Assignment(lform, integr))

		return expr

def main():
	fe_field = [Tet4() ] #, Tet10()]
	fe_test = [Tet4() ] #, Tet10()]

	q = [qx, qy, qz]
	for fe_f in fe_field:
		field = Field(fe_f, coeffs('u', fe_f.n_nodes()))
		for fe_to in fe_test:
			op = GradAndProjectOp(field, fe_to, q)
			print(f'From {fe_f.name()} to {fe_to.name()}')
			c_code(op.apply())


if __name__ == '__main__':
	main()
