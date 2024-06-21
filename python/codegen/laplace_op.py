#!/usr/bin/env python3

from sfem_codegen import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *
from tet20 import *

import sys

class LaplaceOp:
	def __init__(self, fe, symbolic_integration = False):
		self.symbolic_integration = symbolic_integration

		# Ref element dims
		dims = fe.manifold_dim()

		if dims == 1:
			q = [qx]
		elif dims == 2:
			q = [qx, qy]
		else:
			q = [qx, qy, qz]

		# Quadrature point
		q = sp.Matrix(dims, 1, q)

		# Quadrature weight
		qw = sp.symbols('qw')

		ref_grad = fe.grad(q)

		u = coeffs('u', fe.n_nodes())

		J_inv = fe.jacobian_inverse(q)
		# (First Fundamental Form)
		FFF = (J_inv * J_inv.T) * fe.jacobian_determinant(q)

		# We include the reference measure in numerical quadrature
		if not symbolic_integration:
			FFF *= fe.reference_measure()

		###################################################################

		FFF_symbolic = sp.Matrix(dims, dims, [0]*(dims*dims))
		varidx = 0
		for i in range(0, dims):
			for j in range(i, dims):
				var = sp.symbols(f'fff[{varidx}*stride]')
				varidx += 1
				FFF_symbolic[i, j] = var
				FFF_symbolic[j, i] = var;

		###################################################################

		ref_grad_interp = sp.Matrix(dims, 1, [0] * dims)
		for i in range(0, fe.n_nodes()):
			for d in range(0, dims):
				ref_grad_interp[d] += ref_grad[i][d] * u[i]

		trial_operand = FFF_symbolic * ref_grad_interp

		###################################################################

		self.fe = fe
		self.ref_grad = ref_grad
		self.q = q
		self.qw = qw
		self.trial_operand = trial_operand
		self.trial_operand_symbolic = coeffs('trial_operand', dims)
		self.ref_grad_interp = ref_grad_interp
		self.FFF_symbolic = FFF_symbolic
		self.FFF = FFF

	def fff(self):
		expr = []
		for d1 in range(0, self.fe.spatial_dim()):
			for d2 in range(d1,  self.fe.spatial_dim()):
				var = self.FFF_symbolic[d1, d2]
				val = self.FFF[d1, d2]
				expr.append(ast.Assignment(var, val))
		return expr

	def det_fff(self):
		return [determinant(self.FFF_symbolic)]

	def hessian(self):
		fe = self.fe
		ref_grad = self.ref_grad
		FFF_symbolic = self.FFF_symbolic
		fe = self.fe
		q = self.q

		expr = []
		for i in range(0, fe.n_nodes()):
			gi = FFF_symbolic * ref_grad[i]

			for j in range(0, fe.n_nodes()):
				integr = 0

				for d in range(0, fe.manifold_dim()):
					gdotg = gi[d] * ref_grad[j][d]

					if self.symbolic_integration:
						integr += fe.integrate(q, gdotg)
					else:
						integr += gdotg * self.qw

				var = sp.symbols(f'element_matrix[{i*fe.n_nodes() + j}*stride]')
				expr.append(ast.Assignment(var, integr))

		return expr

	def hessian_diag(self):
		fe = self.fe
		ref_grad = self.ref_grad
		FFF_symbolic = self.FFF_symbolic
		fe = self.fe
		q = self.q

		expr = []
		for i in range(0, fe.n_nodes()):
			gi = FFF_symbolic * ref_grad[i]

			integr = 0
			for d in range(0, fe.manifold_dim()):
				gdotg = gi[d] * ref_grad[i][d]

				if self.symbolic_integration:
					integr += fe.integrate(q, gdotg)
				else:
					integr += gdotg * self.qw

			var = sp.symbols(f'element_vector[{i}*stride]')

			if self.symbolic_integration:
				expr.append(ast.Assignment(var, integr))
			else:
				expr.append(ast.AddAugmentedAssignment(var, integr))

		return expr

	def trial_operand_expr(self):
		expr = []
		for d in range(0,  self.fe.spatial_dim()):
			var = self.trial_operand_symbolic[d]
			val = self.trial_operand[d] * self.qw # NOTE that quadrature weight is used here
			expr.append(ast.Assignment(var, val))
		return expr

	def apply(self):
		fe = self.fe
		ref_grad = self.ref_grad
		q = self.q

		if self.symbolic_integration:
			trial_operand = self.FFF_symbolic * self.ref_grad_interp 
		else:
			trial_operand = self.trial_operand_symbolic # NOTE that quadrature weight is included here

		expr = []
		for i in range(0, fe.n_nodes()):
			integr = 0

			for d in range(0, fe.manifold_dim()):
				gdotg = trial_operand[d] * ref_grad[i][d]

				if self.symbolic_integration:
					integr += fe.integrate(q, gdotg)
				else:
					integr += gdotg

			lform = sp.symbols(f'element_vector[{i}*stride]')

			if self.symbolic_integration:
				expr.append(ast.Assignment(lform, integr))
			else:
				expr.append(ast.AddAugmentedAssignment(lform, integr))

		return expr

	def value(self):
		fe = self.fe
		q = self.q

		if self.symbolic_integration:
			trial_operand = self.FFF_symbolic * self.ref_grad_interp
		else:
			trial_operand = self.trial_operand_symbolic  # NOTE that quadrature weight is included here

		integr = 0
		for d in range(0, fe.manifold_dim()):
			if self.symbolic_integration:
				gsquared = fe.integrate(q, (trial_operand[d] * self.ref_grad_interp[d])) / 2
			else:
				gsquared = trial_operand[d] * self.ref_grad_interp[d] / 2
			integr += gsquared

		integr = sp.simplify(integr)

		form = sp.symbols(f'element_scalar[0]')

		if self.symbolic_integration:
			return [ast.Assignment(form, integr)]
		else:
			return [ast.AddAugmentedAssignment(form, integr)]

def main():

	fes = {
	"TRI6": Tri6(),
	"TRI3": Tri3(),
	"TET4": Tet4(),
	"TET10": Tet10(),
	"TET20": Tet20()}

	if len(sys.argv) >= 2:
		fe = fes[sys.argv[1]]
	else:
		print("Fallback with TET10")
		fe = Tet10()

	symbolic_integration = False
	if len(sys.argv) >= 3:
		symbolic_integration = int(sys.argv[2])

	op = LaplaceOp(fe, symbolic_integration)

	print('---------------------------------------------------')
	print("fff")
	print('---------------------------------------------------')

	c_code(op.fff())

	print('---------------------------------------------------')
	print("det_fff")
	print('---------------------------------------------------')

	c_code(op.det_fff())


	if not symbolic_integration:
		print('---------------------------------------------------')
		print("trial_operand")
		print('---------------------------------------------------')
		c_code(op.trial_operand_expr())

	print('---------------------------------------------------')
	print("hessian")
	print('---------------------------------------------------')

	c_code(op.hessian())

	print('---------------------------------------------------')
	print("apply")
	print('---------------------------------------------------')

	c_code(op.apply())

	print('---------------------------------------------------')
	print("hessian_diag")
	print('---------------------------------------------------')

	c_code(op.hessian_diag())

	print('---------------------------------------------------')
	print("Value")
	print('---------------------------------------------------')

	c_code(op.value())

if __name__ == '__main__':
	main()
