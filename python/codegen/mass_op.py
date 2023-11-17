#!/usr/bin/env python3

from sfem_codegen import *

from tet4 import *
from tet10 import *

from tri3 import *
from tri6 import *

from edge2 import *

from fields import *

class MassOp:
	def __init__(self, field, fe_test, q):
		self.field = field
		self.fe_trial = field.fe
		self.fe_test = fe_test
		self.q = q

	def matrix(self):
		fe_trial = self.fe_trial
		fe_test = self.fe_test
		q = self.q

		fun_trial = fe_trial.fun(q)
		fun_test  = fe_test.fun(q)
		dV = fe_test.jacobian_determinant(q)

		expr = []
		for i in range(0, fe_test.n_nodes()):
			for j in range(0, fe_trial.n_nodes()):
				integr = fe_test.integrate(q, fun_test[i] * fun_trial[j]) * dV
				var = sp.symbols(f'element_matrix[{i*fe_trial.n_nodes()+j}]')
				expr.append(ast.Assignment(var, sp.simplify(integr)))

		return expr

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
			integr = sp.simplify(integr)
			d[i] = integr
			approx_measure += integr

		measure = fe_test.measure(q)
		rescale = measure / approx_measure
		d = d * rescale
		
		for i in range(0, fe_test.n_nodes()):
			var = sp.symbols(f'element_matrix_diag[{i}]')
			expr.append(ast.Assignment(var, d[i]))
		return expr

	# def biorthogonal_weights(self):
	# 	mmat = self.matrix()
	# 	mvec = self.hrz_diagonal_scaling_lumped_matrix()


	def lumped_matrix(self):
		fe_trial = self.fe_trial
		fe_test = self.fe_test
		q = self.q

		fun_trial = fe_trial.fun(q)
		fun_test  = fe_test.fun(q)
		dV = fe_test.jacobian_determinant(q)

		expr = []
		for i in range(0, fe_test.n_nodes()):
			integr = 0
			for j in range(0, fe_trial.n_nodes()):
				integr += fe_test.integrate(q, fun_test[i] * fun_trial[j]) * dV
			
			var = sp.symbols(f'element_matrix_diag[{i}]')
			expr.append(ast.Assignment(var, sp.simplify(integr)))

		return expr

	def apply(self):
		field = self.field
		fe_trial = self.fe_trial
		fe_test = self.fe_test
		q = self.q

		fun_trial = field.eval(q)
		fun_test  = fe_test.fun(q)
		dV = fe_test.jacobian_determinant(q)

		expr = []
		for i in range(0, fe_test.n_nodes()):
			integr = fe_test.integrate(q, fun_test[i] * fun_trial) * dV

			integr = sp.simplify(integr)
			var = sp.symbols(f'element_vector[{i}]')
			expr.append(ast.Assignment(var, sp.simplify(integr)))

		return expr

def main():

	if True:
		fe = Beam2()
		f =  Field(fe, coeffs('u', 2))
		op = MassOp(f, fe, [qx])
		c_code(op.apply())
		# c_code(op.lumped_matrix())
		# c_code(op.matrix())

	if False:
		fe = Tri3()
		f =  Field(fe, coeffs('u', 3))
		op = MassOp(f, fe, [qx, qy])
		c_code(op.apply())

	if False:
		fe = TriShell3()
		f =  Field(fe, coeffs('u', 3))
		op = MassOp(f, fe, [qx, qy])
		# c_code(op.apply())
		# c_code(op.lumped_matrix())
		c_code(op.matrix())

	if False:
		fe = Tet10()
		f =  Field(fe, coeffs('u', 10))
		op = MassOp(f, fe, [qx, qy, qz])
		# c_code(op.matrix())
		# c_code(op.hrz_diagonal_scaling_lumped_matrix())
		c_code(op.lumped_matrix())

	if False:
		fe = Tri6()
		f =  Field(fe, coeffs('u', 6))
		op = MassOp(f, fe, [qx, qy, qz])
		# c_code(op.matrix())
		c_code(op.hrz_diagonal_scaling_lumped_matrix())
		# c_code(op.lumped_matrix())

	if False:
	# if True:
		# fields = [Field(TriShell3(), coeffs('u', 3)), Field(TransformedTet10(), coeffs('u', 10))]
		# test_fes  = [TransformedTet10(), DualTet10()]

		# fields = [Field(TriShell3(), coeffs('u', 3)), Field(TransformedTriShell6(), coeffs('u', 6))]
		# test_fes  = [TransformedTriShell6(), DualTriShell6()]

		# fields = [Field(Tri6(), coeffs('u', 6)), Field(TransformedTri6(), coeffs('u', 6))]
		# test_fes  = [TransformedTri6(), DualTri6()]

		# fields = [Field(Tri6(), coeffs('u', 6))]
		# test_fes  = [Tri6()]

		fields = [Field(Tri3(), coeffs('u', 3))]
		test_fes  = [Tri3()]

		n_forms = len(test_fes)

		q = [qx, qy, qz]
		for i in range(0, n_forms):
			trial = fields[i]
			test = test_fes[i]
			op = MassOp(trial, test, q)
			expr = op.apply()

			print("----------------------------")
			print(f"MassOp({trial.fe.name()}, {test.name()})")
			print("----------------------------")
			print("Eval")
			print("----------------------------")
			c_code(expr)
			print("----------------------------")

			expr = op.lumped_matrix()

			print("----------------------------")
			print("lumped Matrix")
			print("----------------------------")
			c_code(expr)
			print("----------------------------")	


			print("----------------------------")
			print("Matrix")
			print("----------------------------")
			expr = op.matrix()
			c_code(expr)
			print("----------------------------")	

			# expr = op.matrix()

			# print("----------------------------")
			# print("Matrix")
			# print("----------------------------")
			# c_code(expr)
			# print("----------------------------")	

	# if True:
	if False:
		print("----------------------------")
		print("Transform")
		print("----------------------------")
		c_code(tet10_basis_transform_expr())
		print("----------------------------")

	if False:
	# if False:
		print("----------------------------")
		print("Transform")
		print("----------------------------")
		c_code(tri6_basis_transform_expr())
		print("----------------------------")

if __name__ == '__main__':
	main()
