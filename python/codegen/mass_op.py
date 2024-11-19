#!/usr/bin/env python3

# References
# [1] Zienkiewicz, O.C., Taylor, R.L. and Zhu, J.Z., 2005. 	The finite element method: its basis and fundamentals. Elsevier.
# [2] Duczek, S. and Gravenkamp, H., 2019. 					Critical assessment of different mass lumping schemes for higher order serendipity finite elements. Computer Methods in Applied Mechanics and Engineering, 350, pp.836-897.
# [3] Hinton, E., Rock, T. and Zienkiewicz, O.C., 1976. 	A note on mass lumping and related processes in the finite element method. Earthquake Engineering & Structural Dynamics, 4(3), pp.245-249.

from sfem_codegen import *

from tet4 import *
from tet10 import *
from tet20 import *

from hex8 import *
from aahex8 import *

from tri3 import *
from tri6 import *
from quad4 import *

from edge2 import *

from fields import *

import sys

def simplify(expr):
	# return sp.simplify(expr)
	return expr

class MassOp:
	def __init__(self, field, fe_test, symbolic_integration):
		# Ref element dims
		dims = fe_test.manifold_dim()

		# Quadrature point
		q = fe_test.quadrature_point()

		self.field = field
		self.fe_trial = field.fe
		self.fe_test = fe_test
		self.q = q
		self.qw = fe_test.quadrature_weight()
		self.symbolic_integration = symbolic_integration

	def sym_matrix(self):
		fe_trial = self.fe_trial
		fe_test = self.fe_test
		q = self.q

		fun_trial = fe_trial.fun(q)
		fun_test  = fe_test.fun(q)

		if fe_test.is_isoparametric():
			dV = fe_test.jacobian_determinant(q)
		else:
			dV = fe_test.symbol_jacobian_determinant()

		if not self.symbolic_integration:
			dV *= fe_test.reference_measure() * self.qw

		M = sp.zeros(fe_test.n_nodes(), fe_trial.n_nodes())
		for i in range(0, fe_test.n_nodes()):
			for j in range(0, fe_trial.n_nodes()):
				if not self.symbolic_integration:
					integr = fun_test[i] * fun_trial[j] * dV
				else:
					if fe_test.is_isoparametric():
						integr = fe_test.integrate(q, fun_test[i] * fun_trial[j] * dV) 
					else:
						integr = fe_test.integrate(q, fun_test[i] * fun_trial[j]) * dV
				M[i, j] = integr

		return M

	def matrix(self):
		M = self.sym_matrix()
		rows, cols = M.shape

		expr = []
		for i in range(0, rows):
			for j in range(0, cols):
				var = sp.symbols(f'element_matrix[{i*cols+j}]')
				expr.append(self.assign_op(var, M[i, j]))

		return expr

	# [3]
	def hrz_diagonal_scaling_lumped_matrix(self):
		fe_trial = self.fe_trial
		fe_test = self.fe_test
		q = self.q

		fun_trial = fe_trial.fun(q)
		fun_test  = fe_test.fun(q)
		dV = fe_test.jacobian_determinant(q)

		expr = []

		d = sp.zeros(fe_test.n_nodes(), 1)

		approx_measure = 0
		for i in range(0, fe_test.n_nodes()):
			integr = fe_test.integrate(q, fun_test[i] * fun_trial[i]) * dV
			integr = simplify(integr)
			d[i] = integr
			approx_measure += integr

		measure = fe_test.measure(q)
		rescale = measure / approx_measure
		d = d * rescale
		
		for i in range(0, fe_test.n_nodes()):
			var = sp.symbols(f'element_matrix_diag[{i}]')
			expr.append(ast.Assignment(var, d[i]))
		return expr

	def lumped_matrix(self):
		fe_trial = self.fe_trial
		fe_test = self.fe_test
		q = self.q

		fun_trial = fe_trial.fun(q)
		fun_test  = fe_test.fun(q)
		dV = fe_test.jacobian_determinant(q)

		if not self.symbolic_integration:
			dV *= fe_test.reference_measure() * self.qw

		expr = []
		for i in range(0, fe_test.n_nodes()):
			integr = 0
			for j in range(0, fe_trial.n_nodes()):
				if not self.symbolic_integration:
					integr += fun_test[i] * fun_trial[j] * dV
				else:
					if fe_test.is_isoparametric():
						integrand = fun_test[i] * fun_trial[j] * dV
						integrand = simplify(integrand)
						integr += fe_test.integrate(q, integrand) 
					else:
						integr += fe_test.integrate(q, fun_test[i] * fun_trial[j]) * dV

			var = sp.symbols(f'element_matrix_diag[{i}]')
			expr.append(self.assign_op(var, integr))

		return expr

	def apply(self):
		field = self.field
		fe_trial = self.fe_trial
		fe_test = self.fe_test
		q = self.q

		fun_trial = field.eval(q)
		fun_test  = fe_test.fun(q)
		dV = fe_test.jacobian_determinant(q)

		if not self.symbolic_integration:
			dV *= fe_test.reference_measure() * self.qw

		expr = []
		for i in range(0, fe_test.n_nodes()):
			if not self.symbolic_integration:
				integr = fun_test[i] * fun_trial  * dV
			else:
				if fe_test.is_isoparametric():
					integr = fe_test.integrate(q, fun_test[i] * fun_trial  * dV)
				else:
					integr = fe_test.integrate(q, fun_test[i] * fun_trial) * dV

			var = sp.symbols(f'element_vector[{i}]')
			expr.append(self.assign_op(var, integr))
		return expr

	def assign_op(self, var, expr):
		if self.symbolic_integration:
			return ast.Assignment(var, simplify(expr))
		else:
			return ast.AddAugmentedAssignment(var, simplify(expr))

	def apply_to_constant(self):
		field = self.field
		fe_test = self.fe_test
		q = self.q

		fun_test = fe_test.fun(q)
		dV = fe_test.jacobian_determinant(q)
		val = sp.symbols('val')

		if not self.symbolic_integration:
			dV *= fe_test.reference_measure() * self.qw

		expr = []
		for i in range(0, fe_test.n_nodes()):
			if not self.symbolic_integration:
				integr = fun_test[i] * val * dV
			else:
				if fe_test.is_isoparametric():
					integrand = fun_test[i] * val * dV
					integrand = simplify(integrand)
					integr = fe_test.integrate(q, integrand)
				else:
					integr = fe_test.integrate(q, fun_test[i] * val) * dV

			var = sp.symbols(f'element_vector[{i}]')
			expr.append(self.assign_op(var, integr))
		return expr

def main():
	fes = {
		"TRI6": Tri6(),
		"TRI3": Tri3(),
		"TRISHELL3": TriShell3(),
		"TET4": Tet4(),
		"TET10": Tet10(),
		"TET20": Tet20(),
		"HEX8": Hex8(),
		"AAHEX8": AAHex8(),
		"AAQUAD4": AxisAlignedQuad4(),
		"QUAD4": Quad4(),
		"QUADSHELL4": QuadShell4()
	}

	symbolic_integration = False

	if len(sys.argv) >= 2:
		fe = fes[sys.argv[1]]
	else:
		print("Fallback with TET10")
		fe = Tet10()

	fe_field = fe
	if len(sys.argv) >= 3:
		fe_field = fes[sys.argv[2]]

	if len(sys.argv) >= 4:
		symbolic_integration = int(sys.argv[3])

	f = Field(fe_field, coeffs('u', fe_field.n_nodes()))
	op = MassOp(f, fe, symbolic_integration)

	print("----------------------------")
	print("apply_to_constant")
	print("----------------------------")
	c_code(op.apply_to_constant())
	print("----------------------------")	

	print("----------------------------")
	print("lumped_matrix")
	print("----------------------------")
	c_code(op.lumped_matrix())
	print("----------------------------")	

	print("----------------------------")
	print("matrix")
	print("----------------------------")
	c_code(op.matrix())
	print("----------------------------")	

	print("----------------------------")
	print("apply")
	print("----------------------------")
	c_code(op.apply())
	print("----------------------------")	

if __name__ == '__main__':
	main()
