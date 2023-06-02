#!/usr/bin/env python3

from sfem_codegen import *
from quad4 import *

class LinearElasticityOp:
	def __init__(self, fe, q):
		dims = fe.manifold_dim()
		q = sp.Matrix(dims, 1, q)
		shape_grad = fe.physical_tgrad(q)
		e_jac_inv = fe.jacobian_inverse(q)
		dV = fe.jacobian_determinant(q)
		s_jac_inv = fe.symbol_jacobian_inverse()
		disp = coeffs('u', dims * fe.n_nodes())

		rows = fe.n_nodes() * dims
		cols = rows

		###################################################################
		# Material law
		###################################################################

		mu, lmbda = sp.symbols('mu lambda', real=True)
		s_disp_grad = sp.Matrix(dims, dims, coeffs('disp_grad', dims * dims))
		epsu = (s_disp_grad + s_disp_grad.T) / 2
		e_disp_grad = sp.Matrix(dims, dims, [0] * dims * dims)

		for i in range(0, dims * fe.n_nodes()):
			e_disp_grad += disp[i] * shape_grad[i]

		# strain energy function
		e = lmbda/2 * tr(epsu) * tr(epsu) + mu * inner(epsu, epsu)

		# Gradient
		de = sp.Matrix(dims, dims, [0]*(dims*dims))
		for d1 in range(0, dims):
			for d2 in range(0, dims):
				de[d1, d2] = sp.diff(e, s_disp_grad[d1, d2])

		eval_grad = sp.Matrix(rows, 1, [0] * rows)
		for i in range(0, fe.n_nodes() * dims):
			eval_grad[i] = inner(de, shape_grad[i])

		dde = sp.Matrix(dims, dims, [0]*(dims*dims))
		eval_hessian =  sp.Matrix(rows, cols, [0] * (rows * cols))

		for i in range(0, rows):
			for d1 in range(0, dims):
				for d2 in range(0, dims):
					dde[d1, d2] = sp.diff(eval_grad[i], s_disp_grad[d1, d2])

			for j in range(0, cols):
				eval_hessian[i, j] = inner(dde, shape_grad[j])

		###################################################################
		# Integrate and subsitute
		###################################################################

		integr_grad = sp.Matrix(rows, 1, [0] * rows)
		integr_hessian = sp.Matrix(rows, cols, [0] * (rows * cols))

		for i in range(0, rows):
			integr = fe.integrate(q, eval_grad[i])
			integr = subsmat(integr, s_disp_grad, e_disp_grad)
			integr = subsmat(integr, s_jac_inv, e_jac_inv)
			integr = integr * dV
			integr_grad[i] = integr

		for i in range(0, rows):
			for j in range(0, cols):
				integr = fe.integrate(q, eval_hessian[i, j])
				integr = subsmat(integr, s_disp_grad, e_disp_grad)
				integr = subsmat(integr, s_jac_inv, e_jac_inv)
				integr = integr * dV
				integr_hessian[i, j] = integr

		###################################################################

		self.e = e
		self.de = de
		self.dde = dde
		self.mu = mu
		self.lmbda = lmbda

		self.eval_grad = eval_grad
		self.eval_hessian = eval_hessian
		self.integr_grad = integr_grad
		self.integr_hessian = integr_hessian

		###################################################################

		self.fe = fe

		###################################################################
	def hessian_check(self):
		H = self.integr_hessian
		rows, cols = H.shape

		A = sp.Matrix(rows, cols, [0] * (rows * cols))
		for i in range(0, rows):
			for j in range(0, cols):
				integr = H[i, j]
				integr = integr.subs(self.mu, 2)
				integr = integr.subs(self.lmbda, 2)

				integr = integr.subs(x0, 0)
				integr = integr.subs(y0, 0)
				
				integr = integr.subs(x1, 0.5)
				integr = integr.subs(y1, 0)

				integr = integr.subs(x2, 0.5)
				integr = integr.subs(y2, 0.5)

				integr = integr.subs(x3, 0)
				integr = integr.subs(y3, 0.5)
				
				A[i, j] = integr
		c_log(A)

	def hessian(self):
		H = self.integr_hessian
		rows, cols = H.shape

		expr = []
		for i in range(0, rows):
			for j in range(0, cols):
				var = sp.symbols(f'element_matrix[{i*cols + j}*stride]')
				expr.append(ast.Assignment(var, H[i, j]))

		return expr

	def gradient(self):
		fe = self.fe
		expr = []
		# for i in range(0, fe.n_nodes()):
		# 	integr = 0

		# 	for d in range(0, fe.manifold_dim()):
		# 		gdotg = FFF_x_g[d] * g[i][d]
		# 		integr += fe.integrate(q, gdotg)

		# 	lform = sp.symbols(f'element_vector[{i}*stride]')
		# 	expr.append(ast.Assignment(lform, integr))

		# pdb.set_trace()
		return expr

	def value(self):
		fe = self.fe
		expr = []

		
		# for d in range(0, fe.manifold_dim()):
		# 	gsquared = fe.integrate(q, (grad_uh[d] **2)) / 2
		# 	integr += gsquared


		# for d1 in range(0, fe.manifold_dim()):
		# 	for d2 in range(0, fe.manifold_dim()):
		# 		integr = integr.subs(s_jac_inv[d1, d2], e_jac_inv[d1, d2])

		# integr *= dV

		# form = sp.symbols(f'element_scalar[0]')
		# expr.append(ast.Assignment(form, integr))
		return expr

def main():
	fe = AxisAlignedQuad4()
	q = sp.Matrix(2, 1, [qx, qy])
	op = LinearElasticityOp(fe, q)
	op.hessian_check()
	# c_code(op.hessian())

if __name__ == '__main__':
	main()