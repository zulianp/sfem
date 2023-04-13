#!/usr/bin/env python3

from sfem_codegen import *

from tet4 import *
from tri3 import *
from tri6 import *
from tri1 import *

from fields import *

class GradAndProjectOp:
	def __init__(self, fe_from, fe_to, q):
		self.q = sp.Matrix(fe_from.manifold_dim(), 1, q)
		self.u_from = coeffs('u', fe_from.n_nodes())
		self.fe_from = fe_from
		self.fe_to = fe_to

	def matrix(self):
		# fe_from = self.fe_from
		# fe_to = self.fe_to
		# q = self.q

		# f_from = fe_from.fun(q)
		# f_to   = fe_to.fun(q)

		# expr = []
		# for i in range(0, fe_to.n_nodes()):
		# 	for j in range(0, fe_from.n_nodes()):
		# 		integr = fe_to.integrate(q, f_from[j] * f_to[i] * fe_to.jacobian_determinant(q))
		# 		var = sp.symbols(f'element_matrix[{i*fe.n_nodes() + j}*stride]')
		# 		expr.append(ast.Assignment(var, integr))

		# return expr

	def apply(self):
		# fe_from = self.fe_from
		# fe_to = self.fe_to
		# q = self.q

		# f_from = fe_from.fun(q)
		# f_to   = fe_to.fun(q)

		# u_from = self.u_from

		# uh = 0
		# for i in range(0, fe_from.n_nodes()):
		# 	uh += u_from[i] * f_from[i]

		# expr = []
		# for i in range(0, fe_to.n_nodes()):
		# 	integr = fe_to.integrate(q, uh * f_to[i] * fe_to.jacobian_determinant(q))
		# 	lform = sp.symbols(f'element_vector[{i}*stride]')
		# 	expr.append(ast.Assignment(lform, integr))
		# return expr

def main():
	shell_fe_from = [Tet4(), Tet10()]
	shell_fe_to = [Tet4(), Tet10()]

	q = [qx, qy, qz]
	for sf_from in shell_fe_from:
		for sf_to in shell_fe_to:
			op = GradAndProjectOp(sf_from, sf_to, q)
			print(f'From {sf_from.name()} to {sf_to.name()}')
			c_code(op.apply())


if __name__ == '__main__':
	main()
