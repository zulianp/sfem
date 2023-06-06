#!/usr/bin/env python3

from sfem_codegen import *
from quad4 import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *

from time import perf_counter

# AT2
def g(c):
	return (1 - c)**2

def omega(c):
	return c**2

def comega(c):
 	return 2

class PhaseFieldForFractureOp:
	def __init__(self, fe, q):
		dims = fe.manifold_dim()
		q = sp.Matrix(dims, 1, q)
		shape_fun = fe.fun(q)
		shape_grad = fe.physical_grad(q)
		tensor_shape_grad = fe.physical_tgrad(q)
		e_jac_inv = fe.jacobian_inverse(q)
		dV = fe.jacobian_determinant(q)

		s_jac_inv = fe.symbol_jacobian_inverse()
		disp = coeffs('u', dims * fe.n_nodes())
		c = coeffs('c', fe.n_nodes())

		n_var = dims + 1
		rows = fe.n_nodes() * n_var
		cols = rows
		
		###################################################################
		# FE function and evaluations
		###################################################################
		s_disp_grad = sp.Matrix(dims, dims, coeffs('disp_grad', dims * dims))
		e_disp_grad = sp.Matrix(dims, dims, [0] * dims * dims)

		for i in range(0, dims * fe.n_nodes()):
			e_disp_grad += disp[i] * tensor_shape_grad[i]

		s_c = sp.symbols('c', real=True)
		s_gradc = coeffs('gradc', dims)

		e_c = 0
		e_gradc =  sp.Matrix(dims, 1, [0] * dims)

		for i in range(0, fe.n_nodes()):
			e_c += c[i] * shape_fun[i]
			e_gradc += c[i] * shape_grad[i]

		###################################################################
		# Material law
		###################################################################
		c_log("Material law")

		Gc, ls = sp.symbols('Gc ls', real=True)
		mu, lmbda = sp.symbols('mu lambda', real=True)
		epsu = (s_disp_grad + s_disp_grad.T) / 2
		
		# strain energy function
		eu = mu * inner(epsu, epsu) + (lmbda/2) * (tr(epsu) * tr(epsu))
		ec = (Gc / comega(s_c)) * (omega(s_c)/ls + ls * inner(s_gradc, s_gradc))
		e = g(s_c) * eu + ec

		###################################################################
		# Gradient
		###################################################################
		eval_grad = sp.Matrix(rows, 1, [0] * rows)

		# Phase-field
		dedc = sp.diff(e, s_c)
		dedgradc = sp.diff(e, s_gradc)

		# Displacement
		dedu = sp.Matrix(dims, dims, [0]*(dims*dims))
		for d1 in range(0, dims):
			for d2 in range(0, dims):
				dedu[d1, d2] = sp.diff(e, s_disp_grad[d1, d2])

		for i in range(0, fe.n_nodes()):
			eval_grad[i] = dedc * shape_fun[i]
			eval_grad[i] += inner(dedgradc, shape_grad[i])

		for i in range(0, fe.n_nodes() * dims):
			eval_grad[fe.n_nodes() + i] = inner(dedu, tensor_shape_grad[i])

		###################################################################
		# Hessian
		###################################################################
		eval_hessian =  sp.Matrix(rows, cols, [0] * (rows * cols))

		# Phase-field

		# Loop over all variables
		for j in range(0, cols):
			ddeddc = sp.diff(eval_grad[j], s_c)
			ddeddgradc = sp.diff(eval_grad[j], s_gradc)

			for i in range(0, fe.n_nodes()):
				eval_hessian[i, j] = ddeddc * shape_fun[i] + inner(ddeddgradc, shape_grad[i])


		# Displacement
		ddedu = sp.Matrix(dims, dims, [0]*(dims*dims))
		
		# Loop over all variables
		for j in range(0, cols):
			for d1 in range(0, dims):
				for d2 in range(0, dims):
					ddedu[d1, d2] = sp.diff(eval_grad[j], s_disp_grad[d1, d2])

			for i in range(0, fe.n_nodes() * dims):
				eval_hessian[fe.n_nodes() + i, j] = inner(ddedu, tensor_shape_grad[i])

		###################################################################
		# Integrate and substitute
		###################################################################
		c_log("Integrate")
		full_eval = True

		integr_value = 0
		integr_gradient = sp.Matrix(rows, 1, [0] * rows)
		integr_hessian = sp.Matrix(rows, cols, [0] * (rows * cols))
		integr_value = fe.integrate(q, e)

		if full_eval:
			integr_value = subsmat(integr_value, s_disp_grad, e_disp_grad)
			integr_value = subsmat(integr_value, s_jac_inv, e_jac_inv)
			integr_value = integr_value.subs(s_c, e_c)
			integr_value = subsmat(integr_value, s_gradc, e_gradc)

		integr_value = integr_value * dV

		for i in range(0, rows):
			integr = fe.integrate(q, eval_grad[i])

			if full_eval:
				integr = subsmat(integr, s_disp_grad, e_disp_grad)
				integr = subsmat(integr, s_jac_inv, e_jac_inv)
				integr = integr.subs(s_c, e_c)
				integr = subsmat(integr, s_gradc, e_gradc)

			integr = integr * dV
			integr_gradient[i] = integr

		for i in range(0, rows):
			for j in range(0, cols):
				integr = fe.integrate(q, eval_hessian[i, j])

				if full_eval:
					integr = subsmat(integr, s_disp_grad, e_disp_grad)
					integr = subsmat(integr, s_jac_inv, e_jac_inv)
					integr = integr.subs(s_c, e_c)
					integr = subsmat(integr, s_gradc, e_gradc)

				integr = integr * dV
				integr_hessian[i, j] = integr

		###################################################################

		self.e = e
		self.mu = mu
		self.lmbda = lmbda
		self.disp = disp
		self.c = c
		self.Gc = Gc
		self.ls = ls

		self.eval_grad = eval_grad
		self.eval_hessian = eval_hessian

		self.integr_value = integr_value
		self.integr_gradient = integr_gradient
		self.integr_hessian = integr_hessian

		self.increment = coeffs('increment', fe.n_nodes() * n_var)

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

				coord = 1
				integr = integr.subs(self.mu, 0.5)
				integr = integr.subs(self.lmbda, 1)
				integr = integr.subs(self.ls, 1)
				integr = integr.subs(self.Gc, 1)

				for c in self.c:
					integr = integr.subs(c, 0)

				for u in self.disp:
					integr = integr.subs(u, 0)

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
		var = sp.symbols(f'element_scalar[0]')
		return [ast.Assignment(var, self.integr_value)]


	def apply(self):
		H = self.integr_hessian
		rows, cols = H.shape
		inc = self.increment

		expr = []
		for i in range(0, rows):
			val = 0
			for j in range(0, cols):
				val += H[i, j] * inc[j]

			var = sp.symbols(f'element_vector[{i}*stride]')
			expr.append(ast.Assignment(var, val))
		return expr


def main():
	start = perf_counter()
	fe = AxisAlignedQuad4()

	# fe = Tri3()
	# fe = Tri6()
	q = sp.Matrix(2, 1, [qx, qy])

	# fe = Tet4()
	# fe = Tet10()
	# q = sp.Matrix(3, 1, [qx, qy, qz])

	op = PhaseFieldForFractureOp(fe, q)
	op.hessian_check()

	if False:
	# if True:
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

		c_log("--------------------------")
		c_log("apply")	
		c_log("--------------------------")
		c_code(op.apply())

	stop = perf_counter()
	console.print(f'Overall: {stop - start} seconds')


if __name__ == '__main__':
	main()