#!/usr/bin/env python3

from sfem_codegen import *
from quad4 import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *
from tet20 import *
from hex8 import *
from aahex8 import *
from symbolic_fe import *

import sys
from time import perf_counter


class LinearElasticityOpSymbolic:
	def __init__(self, fe):
		trial = "trial"
		test = "test"
		disp_grad_name = 'disp_grad'

		dims = fe.manifold_dim()
		dV = fe.symbol_jacobian_determinant()
		J_inv = fe.symbol_jacobian_inverse_as_adjugate()
		q = fe.quadrature_point()
		qw = fe.quadrature_weight()

		point = coeffs('p', dims)
		
	
		trial_grad = fe.trial_grad(q, dims)
		test_grad  = fe.test_grad(q, dims)

		# Elasticity parameters
		disp_grad = sp.Matrix(dims, dims, coeffs(disp_grad_name, dims * dims))
		epsu = (disp_grad + disp_grad.T) / 2
		mu, lmbda = sp.symbols('mu lambda', real=True)

		# Strain energy function
		e = mu * inner(epsu, epsu) + (lmbda/2) * (tr(epsu) * tr(epsu))

		# Gradient
		P = sp.zeros(dims, dims)
		for d1 in range(0, dims):
			for d2 in range(0, dims):
				P[d1, d2] = sp.diff(e, disp_grad[d1, d2])

		de = sp.zeros(dims, 1)
		for i in range(0, dims):
			de[i] = inner(P * J_inv.T, test_grad[i])

		# Hessian
		dde = sp.zeros(dims, dims)

		for i in range(0, dims):
			for j in range(0, dims):
				S_lin = sp.zeros(dims, dims)
				for d1 in range(0, dims):
					for d2 in range(0, dims):
						S_lin[d1, d2] = sp.diff(de[i], disp_grad[d1, d2])
				dde[i, j] = inner(S_lin * J_inv.T, trial_grad[j])


		# print("// Integrating Gradient")		
		# expr = []
		# for i in range(0, dims):
		# 	print(f"({i+1}) / 3")
		# 	eval_gradient = de[i] * dV
		# 	integr = fe.integrate(point, eval_gradient)

		# 	var = sp.symbols(f'element_vector{i}*stride]')
		# 	expr.append(ast.Assignment(var, integr))
		# c_code(expr)
			
		# -------------------------------------------------
		print("// Integrating Hessian")
		H = sp.zeros(dims, dims)
		for i in range(0, dims):
			for j in range(0, dims):
				print(f"// ({i+1}, {j+1}) / (3, 3)")
				eval_hessian = dde[i, j] * (dV * qw)
				H[i, j] = eval_hessian

		# -------------------------------------------------
		# c_log("// Code Hessian * u")
		# u = coeffs('u', dims)
		# expr = []
		# for i in range(0, dims):
		# 	Hu = 0
		# 	for j in range(0, dims):
		# 		Hu += H[i, j] * u[j]

		# 	Hu = sp.simplify(Hu)
		# 	var = sp.symbols(f'element_vector[{i}*stride]')
		# 	expr.append(ast.Assignment(var, Hu))
		# c_code(expr)

		# c_log("// Code Hessian off diagonal")
		# expr = []
		# for i in range(0, dims):
		# 	for j in range(0, dims):
		# 		var = sp.symbols(f'element_matrix[{i*dims + j}*stride]')
		# 		expr.append(ast.Assignment(var, H[i, j]))
		# c_code(expr)

		c_log("// Code Hessian sym")
		expr = []
		d_idx = 0
		for i in range(0, dims):
			for j in range(i, dims):
				var = sp.symbols(f'element_matrix[{d_idx}*stride]')
				expr.append(ast.Assignment(var, H[i, j]))
				d_idx += 1
		c_code(expr)

		# c_log("// Code Hessian diagonal")
		# expr = []
		# for i in range(0, dims):
		# 	for j in range(0, dims):
		# 		var = sp.symbols(f'element_matrix[{i*dims + j}*stride]')
				
		# 		for d0 in range(0, dims):
		# 			for d1 in range(0, dims):
		# 				for d2 in range(0, dims):
		# 					H[i, j] = sp.simplify(H[i, j].subs(trial_grad[d0][d1, d2], test_grad[d0][d1, d2]))
		# 		expr.append(ast.Assignment(var, H[i, j]))
		# c_code(expr)


class LinearElasticityOpTaylor:	
	def __init__(self, fe):
		trial = "trial"
		test = "test"
		disp_grad_name = 'disp_grad'

		dims = fe.manifold_dim()
		dV = fe.symbol_jacobian_determinant()
		J_inv = fe.symbol_jacobian_inverse_as_adjugate()

		point = coeffs('p', dims)
		c = fe.barycenter()
		order = -1
		# order = 1
	
		trial_grad = fe.taylor_tgrad_symbolic(trial, c, point, order)
		test_grad  = fe.taylor_tgrad_symbolic(trial, c, point, order)

		# Elasticity parameters
		disp_grad = sp.Matrix(dims, dims, coeffs(disp_grad_name, dims * dims))
		epsu = (disp_grad + disp_grad.T) / 2
		mu, lmbda = sp.symbols('mu lambda', real=True)

		# Strain energy function
		e = mu * inner(epsu, epsu) + (lmbda/2) * (tr(epsu) * tr(epsu))

		# Gradient
		P = sp.zeros(dims, dims)
		for d1 in range(0, dims):
			for d2 in range(0, dims):
				P[d1, d2] = sp.diff(e, disp_grad[d1, d2])

		de = sp.zeros(dims, 1)
		for i in range(0, dims):
			de[i] = inner(P * J_inv.T, test_grad[i])

		# Hessian
		dde = sp.zeros(dims, dims)

		for i in range(0, dims):
			for j in range(0, dims):
				S_lin = sp.zeros(dims, dims)
				for d1 in range(0, dims):
					for d2 in range(0, dims):
						S_lin[d1, d2] = sp.diff(de[i], disp_grad[d1, d2])
				dde[i, j] = inner(S_lin * J_inv.T, trial_grad[j])


		# print("// Integrating Gradient")		
		# expr = []
		# for i in range(0, dims):
		# 	print(f"({i+1}) / 3")
		# 	eval_gradient = de[i] * dV
		# 	integr = fe.integrate(point, eval_gradient)

		# 	var = sp.symbols(f'element_vector{i}*stride]')
		# 	expr.append(ast.Assignment(var, integr))
		# c_code(expr)
			
		# -------------------------------------------------
		print("// Integrating Hessian")
		H = sp.zeros(dims, dims)
		for i in range(0, dims):
			for j in range(i, dims):
				print(f"// ({i+1}, {j+1}) / (3, 3)")
				eval_hessian = dde[i, j] * dV
				integr = fe.integrate(point, eval_hessian)
				H[i, j] = integr
				H[j, i] = integr

		# -------------------------------------------------
		# c_log("// Code Hessian * u")
		# u = coeffs('u', dims)
		# expr = []
		# for i in range(0, dims):
		# 	Hu = 0
		# 	for j in range(0, dims):
		# 		Hu += H[i, j] * u[j]

		# 	Hu = sp.simplify(Hu)
		# 	var = sp.symbols(f'element_vector[{i}*stride]')
		# 	expr.append(ast.Assignment(var, Hu))
		# c_code(expr)

		# c_log("// Code Hessian")
		# expr = []
		# for i in range(0, dims):
		# 	for j in range(0, dims):
		# 		var = sp.symbols(f'element_matrix[{i*dims + j}*stride]')
		# 		expr.append(ast.Assignment(var, H[i, j]))
		# c_code(expr)

		c_log("// Code Hessian_sym")
		expr = []
		ii = 0
		for i in range(0, dims):
			for j in range(i, dims):
				var = sp.symbols(f'element_matrix[{ii}*stride]')
				expr.append(ast.Assignment(var, H[i, j]))
				ii += 1
		c_code(expr)

class LinearElasticityOp:
	
	def __init__(self, fe):
		dims = fe.manifold_dim()
		q = fe.quadrature_point()

		shape_grad = fe.physical_tgrad(q)
		e_jac_inv = fe.jacobian_inverse(q)

		if fe.use_adjugate:
			dV = fe.symbol_jacobian_determinant()
		else:
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
		full_eval = False

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

	def hessian(self):
		H = self.integr_hessian
		rows, cols = H.shape

		expr = []
		for i in range(0, rows):
			for j in range(0, cols):
				var = sp.symbols(f'element_matrix[{i*cols + j}*stride]')
				expr.append(ast.Assignment(var, H[i, j]))

		return expr

	def hessian_sym(self):
		H = self.integr_hessian
		rows, cols = H.shape

		expr = []
		idx = 0
		for i in range(0, rows):
			for j in range(0, cols):
				if j > i:
					continue
				var = sp.symbols(f'element_matrix[{idx}*stride]')
				expr.append(ast.Assignment(var, H[i, j]))
				idx += 1

		return expr

	# def hessian_sym_SoA(self):
	# 	H = self.integr_hessian
	# 	rows, cols = H.shape

	# 	expr = []
	# 	fe = self.fe
	# 	dim = fe.spatial_dim()
	# 	n = fe.n_nodes()

	# 	d_idx = 0
	# 	for d1 in range(0, dim):
	# 		for d2 in range(d1, dim):
	# 			idx = 0
	# 			for i in range(0, n):
	# 				for j in range(i, n):
	# 					var = sp.symbols(f'element_matrix_{d_idx}[{idx}]')
	# 					expr.append(ast.Assignment(var, H[d1*8 + i, d2*8+j]))
	# 					idx += 1
	# 			d_idx += 1
	# 	return expr

	def hessian_blocks(self):
		H = self.integr_hessian
		rows, cols = H.shape
		fe = self.fe

		n = fe.n_nodes()
		dims = fe.spatial_dim()

		blocks = []

		for d1 in range(0, dims):
			for d2 in range(0, dims):
				expr = []
				for i in range(0, n):
					for j in range(0, n):
						var = sp.symbols(f'element_matrix[{i*n + j}]')
						expr.append(ast.Assignment(var, H[d1 * n + i, d2 * n + j]))
				blocks.append((f'block_{d1}_{d2}', expr))

		return blocks

	def hessian_blocks_tpl(self):
		tpl="""
template<typename scalar_t, typename accumulator_t>
static inline __host__ __device__ void  cu_hex8_linear_elasticity_matrix_{BLOCK_NAME}(
const scalar_t mu,
const scalar_t lambda,
const scalar_t *const SFEM_RESTRICT adjugate,
const scalar_t jacobian_determinant,
accumulator_t *const SFEM_RESTRICT
element_matrix) 
{{
	{CODE}
}}
"""
		return tpl

	def apply_blocks(self):
		H = self.integr_hessian
		rows, cols = H.shape
		fe = self.fe

		n = fe.n_nodes()
		dims = fe.spatial_dim()

		blocks = []

		u = coeffs('in', n)
		for d1 in range(0, dims):
			for d2 in range(0, dims):
				expr = []
				B = sp.zeros(n, n)
				for i in range(0, n):
					for j in range(0, n):
						B[i,j] = H[d1 * n + i, d2 * n + j]
						
					val = B * u

				for i in range(0, n):
					var = sp.symbols(f'out[{i}]')
					expr.append(ast.Assignment(var, val[i]))
					
				blocks.append((f'block_{d1}_{d2}', expr))

		return blocks

	def apply_blocks_tpl(self):
		tpl="""
template<typename scalar_t, typename accumulator_t>
static inline __host__ __device__ void  cu_hex8_linear_elasticity_apply_{BLOCK_NAME}(
const scalar_t mu,
const scalar_t lambda,
const scalar_t *const SFEM_RESTRICT adjugate,
const scalar_t jacobian_determinant,
const scalar_t *const SFEM_RESTRICT in,
accumulator_t *const SFEM_RESTRICT
oout) 
{{
	{CODE}
}}
"""
		return tpl

	def hessian_diag(self):
		H = self.integr_hessian
		rows, cols = H.shape

		expr = []
		for i in range(0, rows):
			var = sp.symbols(f'element_vector[{i}*stride]')
			expr.append(ast.Assignment(var, H[i, i]))

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

	fes = {
	"TRI6": Tri6(),
	"TRI3": Tri3(),
	"TET4": Tet4(),
	"TET10": Tet10(),
	"TET20": Tet20(),
	"HEX8": Hex8(),
	"AAHEX8": AAHex8(),
	"AAQUAD4": AxisAlignedQuad4(),
	"QUAD4": Quad4(),
	"SymbolicFE3D" : SymbolicFE3D()
	}

	if len(sys.argv) >= 2:
		fe = fes[sys.argv[1]]
	else:
		print("Fallback with TET10")
		fe = Tet10()

	fe.use_adjugate = True
	
	# op = LinearElasticityOp(fe)
	op = LinearElasticityOpSymbolic(SymbolicFE3D())
	# op = LinearElasticityOpTaylor(fe)
	# op.hessian_check()


	# tpl = op.hessian_blocks_tpl()
	# blocks = op.hessian_blocks()
	# for k,v in blocks:
	# 	c_log("//--------------------------")
	# 	c_log(f"// hessian {k}")	
	# 	c_log("//--------------------------")
	# 	code = c_gen(v)
	# 	c_log(tpl.format(BLOCK_NAME=k, CODE=code))


	# tpl = op.apply_blocks_tpl()
	# blocks = op.apply_blocks()
	# for k,v in blocks:
	# 	c_log("//--------------------------")
	# 	c_log(f"// apply {k}")	
	# 	c_log("//--------------------------")
	# 	code = c_gen(v)
	# 	c_log(tpl.format(BLOCK_NAME=k, CODE=code))


	# c_log("--------------------------")
	# c_log("value")
	# c_log("--------------------------")
	# c_code(op.value())

	# c_log("--------------------------")
	# c_log("gradient")
	# c_log("--------------------------")
	# c_code(op.gradient())

	# c_log("--------------------------")
	# c_log("hessian")	
	# c_log("--------------------------")
	# c_code(op.hessian())

	# c_log("--------------------------")
	# c_log("hessian_sym_SoA")	
	# c_log("--------------------------")
	# c_code(op.hessian_sym_SoA())

	# c_log("--------------------------")
	# c_log("apply")	
	# c_log("--------------------------")
	# c_code(op.apply())

	# c_log("--------------------------")
	# c_log("hessian_diag")	
	# c_log("--------------------------")
	# c_code(op.hessian_diag())

	stop = perf_counter()
	console.print(f'Overall: {stop - start} seconds')


if __name__ == '__main__':
	main()