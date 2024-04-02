#!/usr/bin/env python3

from sfem_codegen import *
from quad4 import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *

from time import perf_counter

def assign_matrix(name, mat):
	rows, cols = mat.shape
	expr = []
	for i in range(0, rows):
		for j in range(0, cols):
			var = sp.symbols(f'{name}[{i*cols + j}]')
			expr.append(ast.Assignment(var, mat[i, j]))
	return expr


class LinearElasticityOp:
	def __init__(self, fe, q):
		dims = fe.manifold_dim()
		q = sp.Matrix(dims, 1, q)
		shape_grad = fe.physical_tgrad(q)
		ref_shape_grad = fe.tgrad(q)
		e_jac_inv = fe.jacobian_inverse(q)
		dV = fe.jacobian_determinant(q)
		s_jac_inv = fe.symbol_jacobian_inverse()
		disp = coeffs('u', dims * fe.n_nodes())

		self.jac = fe.jacobian(q)

		full_eval = False

		if not full_eval:
			dV = fe.symbol_jacobian_determinant()

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

		self.e_disp_grad = e_disp_grad

		# strain energy function
		e = mu * inner(epsu, epsu) + (lmbda/2) * (tr(epsu) * tr(epsu))

		# Gradient
		de = sp.Matrix(dims, dims, [0]*(dims*dims))
		for d1 in range(0, dims):
			for d2 in range(0, dims):
				de[d1, d2] = sp.diff(e, s_disp_grad[d1, d2])


		# if True: 
		if False: 
			P = matrix_coeff('P', dims, dims)
			eval_grad = sp.Matrix(rows, 1, [0] * rows)
			for i in range(0, fe.n_nodes() * dims):
				eval_grad[i] = inner(P, shape_grad[i])
			self.P = de
		else:
			P = matrix_coeff('JinvXP', dims, dims)
			eval_grad = sp.Matrix(rows, 1, [0] * rows)
			JintXde = s_jac_inv * de
			for i in range(0, fe.n_nodes() * dims):
				eval_grad[i] = inner(P, ref_shape_grad[i])
			self.P = de

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

		if full_eval:
			integr_value = subsmat(integr_value, s_disp_grad, e_disp_grad)
			integr_value = subsmat(integr_value, s_jac_inv, e_jac_inv)
			integr_value = integr_value * dV

		for i in range(0, rows):
			integr = fe.integrate(q, eval_grad[i])
			if full_eval:
				integr = subsmat(integr, s_disp_grad, e_disp_grad)
				integr = subsmat(integr, s_jac_inv, e_jac_inv)
				integr = integr * dV

			integr_gradient[i] = integr

		for i in range(0, rows):
			for j in range(0, cols):
				integr = fe.integrate(q, eval_hessian[i, j])
				if full_eval:
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
		self.increment = coeffs('increment', fe.n_nodes() * dims)

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

	def displacement_gradient(self):
		e_gradu = self.e_disp_grad
		expr = assign_matrix('disp_grad', e_gradu)
		return expr

	def first_piola(self):
		P = self.P 
		expr = assign_matrix('P', P)
		return expr

	def JinvXP(self):
		dims = self.fe.manifold_dim()
		P = matrix_coeff('P', dims, dims)
		Jinv = self.fe.symbol_jacobian_inverse()
		expr = Jinv * P * self.fe.symbol_jacobian_determinant()
		return assign_matrix('JinvXP', expr)

	def hessian(self):
		H = self.integr_hessian
		rows, cols = H.shape

		expr = []
		for i in range(0, rows):
			for j in range(0, cols):
				var = sp.symbols(f'element_matrix[{i*cols + j}*stride]')
				expr.append(ast.Assignment(var, H[i, j]))

		return expr

	def geometry(self):
		expr = []
		J = self.jac

		# expr = assign_matrix('jacobian', J)
		J_inv = inverse(J)


		expr.extend(assign_matrix('jacobian_inverse', J_inv))

		J_det = determinant(J)
		expr.append(ast.Assignment(sp.symbols('jacobian_determinant'), J_det))
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

	# fe = AxisAlignedQuad4()
	# fe = Tri3()
	# fe = Tri6()
	# q = sp.Matrix(2, 1, [qx, qy])

	fe = Tet4()
	# fe = Tet10()
	q = sp.Matrix(3, 1, [qx, qy, qz])

	op = LinearElasticityOp(fe, q)
	# op.hessian_check()


	c_log("--------------------------")
	c_log("geometry")	
	c_log("--------------------------")
	c_code(op.geometry())

	c_log("--------------------------")
	c_log("displacement_gradient")	
	c_log("--------------------------")
	c_code(op.displacement_gradient())

	c_log("--------------------------")
	c_log("Piola")	
	c_log("--------------------------")
	c_code(op.first_piola())

	c_log("--------------------------")
	c_log("(Jinv * P)")	
	c_log("--------------------------")

	c_code(op.JinvXP())

	c_log("--------------------------")
	c_log("value")
	c_log("--------------------------")
	c_code(op.value())

	c_log("--------------------------")
	c_log("gradient")
	c_log("--------------------------")
	c_code(op.gradient())

	# c_log("--------------------------")
	# c_log("hessian")	
	# c_log("--------------------------")
	# c_code(op.hessian())

	# c_log("--------------------------")
	# c_log("apply")	
	# c_log("--------------------------")
	# c_code(op.apply())

	

	stop = perf_counter()
	console.print(f'Overall: {stop - start} seconds')


if __name__ == '__main__':
	main()