#!/usr/bin/env python3

from sfem_codegen import *
from quad4 import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *

from time import perf_counter


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
		c_log("Material law")

		mu, lmbda = sp.symbols('mu lambda', real=True)
		s_disp_grad = sp.Matrix(dims, dims, coeffs('disp_grad', dims * dims))
		epsu = (s_disp_grad + s_disp_grad.T) / 2
		e_disp_grad = sp.Matrix(dims, dims, [0] * dims * dims)

		for i in range(0, dims * fe.n_nodes()):
			e_disp_grad += disp[i] * shape_grad[i]

		# strain energy function
		e = mu * inner(epsu, epsu) + (lmbda/2) * (tr(epsu) * tr(epsu))

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

		for j in range(0, rows):
			for d1 in range(0, dims):
				for d2 in range(0, dims):
					dde[d1, d2] = sp.diff(eval_grad[j], s_disp_grad[d1, d2])

			for i in range(0, cols):
				eval_hessian[i, j] = inner(dde, shape_grad[i])

		###################################################################
		# Integrate and substitute
		###################################################################
		c_log("Integrate")

		integr_value = 0
		integr_gradient = sp.Matrix(rows, 1, [0] * rows)
		integr_hessian = sp.Matrix(rows, cols, [0] * (rows * cols))

		integr_value = fe.integrate(q, e)
		integr_value = subsmat(integr_value, s_disp_grad, e_disp_grad)
		integr_value = subsmat(integr_value, s_jac_inv, e_jac_inv)

		for i in range(0, rows):
			integr = fe.integrate(q, eval_grad[i])
			integr = subsmat(integr, s_disp_grad, e_disp_grad)
			integr = subsmat(integr, s_jac_inv, e_jac_inv)
			integr = integr * dV
			integr_gradient[i] = integr

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

		self.integr_value = integr_value
		self.integr_gradient = integr_gradient
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

				coord = 0.5
				integr = integr.subs(self.mu, 2)
				integr = integr.subs(self.lmbda, 2)

				# coord = 1.
				# integr = integr.subs(self.mu, sp.Rational(1, 2))
				# integr = integr.subs(self.lmbda, 1)
				# integr = integr.subs(self.mu, 1)
				# integr = integr.subs(self.lmbda, 0)

				integr = integr.subs(x0, 0)
				integr = integr.subs(y0, 0)
				
				integr = integr.subs(x1, coord)
				integr = integr.subs(y1, 0)

				if rows == 8:
					integr = integr.subs(x2, coord)
					integr = integr.subs(y2, coord)

					integr = integr.subs(x3, 0)
					integr = integr.subs(y3, coord)
				else:
					integr = integr.subs(x2, 0)
					integr = integr.subs(y2, coord)
				
				A[i, j] = integr

		S = A.T - A
		row_sum = sp.Matrix(rows, 1, [0] * rows)
		for i in range(0, rows):
			for j in range(0, cols):
				row_sum[i] += A[i, j]

		# for i in range(0, rows):
		# 	c_log("%.3g" % A[i,i])	

		for i in range(0, rows):
			line = ""
			for j in range(0, rows):
				line += "%.5g " % A[i,j]
			c_log(line)

			
		c_log(S)
		c_log(row_sum)

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
		g = self.integr_gradient
		rows, cols = g.shape

		expr = []
		for i in range(0, rows):
			var = sp.symbols(f'element_vector[{i}*stride]')
			expr.append(ast.Assignment(var, g[i]))

		return expr

	def value(self):
		form = sp.symbols(f'element_scalar[0]')
		return [ast.Assignment(form, self.integr_value)]

def main():
	start = perf_counter()

	# fe = AxisAlignedQuad4()
	# fe = Tri3()
	fe = Tri6()
	q = sp.Matrix(2, 1, [qx, qy])

	# fe = Tet4()
	# fe = Tet10()
	# q = sp.Matrix(3, 1, [qx, qy, qz])

	op = LinearElasticityOp(fe, q)
	# op.hessian_check()

	c_log("--------------------------")
	c_log("value")
	c_log("--------------------------")
	c_code(op.value())

	c_log("--------------------------")
	c_log("gradient")
	c_log("--------------------------")
	c_code(op.gradient())

	c_log("--------------------------")
	c_log("hessian")	
	c_log("--------------------------")
	c_code(op.hessian())

	stop = perf_counter()
	console.print(f'Overall: {stop - start} seconds')


if __name__ == '__main__':
	main()