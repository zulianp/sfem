#!/usr/bin/env python3

from sfem_codegen import *
from fields import *
from tet4 import *
from tet10 import *

class DivOp:
	def __init__(self, vector_field, fe_test, q):
		self.vector_field = vector_field
		self.fe_test = fe_test
		self.q = sp.Matrix(fe_test.manifold_dim(), 1, q)

	def apply(self):
		vector_field = self.vector_field
		fe_test = self.fe_test
		q = self.q

		rf = fe_test.fun(q)
		div_field = vector_field.physical_div(q)
		J_inv_sym = vector_field.fe.symbol_jacobian_inverse()
		J_inv = vector_field.fe.jacobian_inverse(q)

		expr = []
		for i in range(0, fe_test.n_nodes()):
			lform = rf[i] * div_field
			integr = fe_test.integrate(q, lform) * fe_test.jacobian_determinant(q)
			integr = subsmat(integr, J_inv_sym, J_inv)

			var = sp.symbols(f'element_vector[{i}]')
			expr.append(ast.Assignment(var, integr))

		return expr

def main():
	fe = Tet10()
	# fe = Tet4()
	
	u = [ coeffs('ux', fe.n_nodes()), coeffs('uy', fe.n_nodes()), coeffs('uz', fe.n_nodes()) ]
	field = VectorField(fe, u)

	q = vec3(qx, qy, qz)
	op = DivOp(field, fe, q)
	c_code(op.apply())

if __name__ == '__main__':
	main()