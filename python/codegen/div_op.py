#!/usr/bin/env python3

from sfem_codegen import *
from fields import *
from tet4 import *
from tet10 import *
from tri3 import *

class DivOp:
	def __init__(self, vector_field, fe_test, q):
		self.vector_field = vector_field
		self.fe_test = fe_test
		self.q = sp.Matrix(fe_test.manifold_dim(), 1, q)

	def hessian(self):
		fe_trial = self.vector_field.fe
		fe_test = self.fe_test
		q = self.q

		rf = fe_test.fun(q)
		J_inv_sym = fe_test.symbol_jacobian_inverse()
		J_inv = fe_test.jacobian_inverse(q)

		grad_trial = fe_trial.grad(q)

		expr = []
		for i in range(0, fe_test.n_nodes()):
			for j in range(0, fe_trial.n_nodes()):
				# AoS layout
				offset_ij = (i * fe_trial.n_nodes() + j) * fe_trial.spatial_dim()

				gj = J_inv_sym.T * grad_trial[j]

				for d in range(0, fe_trial.spatial_dim()):
					blform = rf[i] * gj[d]
					integr = fe_test.integrate(q, blform) * fe_test.jacobian_determinant(q)
					
					# Only for Tet10 it improves
					if fe_test.n_nodes() == 10:
						integr = sp.simplify(integr)

					integr = subsmat(integr, J_inv_sym, J_inv)

					var = sp.symbols(f'element_matrix[{offset_ij + d}]')
					expr.append(ast.Assignment(var, integr))

		return expr


	# Code for applying the divergence op (div u, q)_L^2
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
			
			# Only for Tet10 it improves
			if fe_test.n_nodes() == 10:
				integr = sp.simplify(integr)

			integr = subsmat(integr, J_inv_sym, J_inv)

			var = sp.symbols(f'element_vector[{i}]')
			expr.append(ast.Assignment(var, integr))

		return expr


	def value(self):
		vector_field = self.vector_field
		fe_test = self.fe_test
		q = self.q

		rf = fe_test.fun(q)
		div_field = vector_field.physical_div(q)
		J_inv_sym = vector_field.fe.symbol_jacobian_inverse()
		J_inv = vector_field.fe.jacobian_inverse(q)

		expr = []

		form = div_field
		integr = fe_test.integrate(q, form) * fe_test.jacobian_determinant(q)
		
		# Only for Tet10 it improves
		if fe_test.n_nodes() == 10:
			integr = sp.simplify(integr)

		integr = subsmat(integr, J_inv_sym, J_inv)

		var = sp.symbols(f'element_value[0]')
		expr.append(ast.Assignment(var, integr))

		return expr


def main():
	
	# fe = Tet10()
	# fe = Tet4()

	# u = [ coeffs('ux', fe.n_nodes()), coeffs('uy', fe.n_nodes()), coeffs('uz', fe.n_nodes()) ]
	# q = vec3(qx, qy, qz)

	fe = Tri3()
	u = [ coeffs('ux', fe.n_nodes()), coeffs('uy', fe.n_nodes()) ]
	q = fe.quadrature_point()
	
	field = VectorField(fe, u)

	
	op = DivOp(field, fe, q)


	c_code(op.apply())


	# c_code(op.value())

if __name__ == '__main__':
	main()