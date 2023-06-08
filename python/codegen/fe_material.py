#!/usr/bin/env python3

from sfem_codegen import *

class Symbol:
	def __init__(self, var, evaluation):
		self.var_ = var
		self.expansion_ = expansion
	
	def var(self):
		return self.var_

	def expansion(self):
		return self.expansion_

class FEFunction:
	def __init__(self, name, fe, ncomp):
		self.name_ = name
		self.fe_ = fe
		self.ncomp_ = ncomp

		self.coeffs_ = coeffs(name, fe.n_nodes() * ncomp)

		self.symbols_value_ = coeffs(f's_{name}', ncomp)
		self.symbols_grad_ = sp.Matrix(fe.spatial_dim(), 1, coeffs(f's_grad_{name}', fe.spatial_dim()))
		
		# Basis functions and grads
		self.symbols_trial_fun_ = coeffs(f's_trial_fun_{name}', ncomp)
		self.symbols_test_fun_  = coeffs(f's_test_fun_{name}', ncomp)

		self.symbols_trial_grad_ = sp.Matrix(fe.spatial_dim(), 1, coeffs(f's_trial_grad_{name}', fe.spatial_dim()))
		self.symbols_test_grad_  = sp.Matrix(fe.spatial_dim(), 1, coeffs(f's_test_grad_{name}', fe.spatial_dim()))

		q = [qx, qy, qz]
		q = q[0:fe.manifold_dim()]

		self.q_ = q
		self.shape_fun_  = fe.tfun(q, ncomp)
		self.shape_grad_ = fe.tgrad(q, ncomp)
		
		# Evaluation
		val = self.coeffs_[0] * self.shape_fun_[0]
		for i in range(1, len(self.shape_fun_)):
			val += self.coeffs_[i] * self.shape_fun_[i]

		self.eval_value_ = val

		val = self.coeffs_[0]  *self.shape_grad_[0]
		for i in range(1, len(self.shape_grad_)):
			val += self.coeffs_[i]  *self.shape_grad_[i]

		self.eval_grad_ = val

	def value(self):
		return Symbol(self.symbols_value_, self.eval_value_)

	def grad(self):
		return Symbol(self.symbols_grad_, self.eval_grad_)

	def fe(self):
		return self.fe_

	def quadrature_point(self):
		return self.q_

	def symbols_value(self):
		return self.symbols_value_ 

	def symbols_grad(self):
		return self.symbols_grad_ 

	def symbols_trial_fun(self):
		return self.symbols_trial_fun_ 

	def symbols_test_fun(self):
		return self.symbols_test_fun_ 

	def symbols_trial_grad(self):
		return self.symbols_trial_grad_ 

	def symbols_test_grad(self):
		return self.symbols_test_grad_ 

	def shape_fun(self):
		return self.shape_fun_ 

	def shape_grad(self):
		return self.shape_grad_ 

	def eval_value(self):
		return self.eval_value_

	def eval_grad(self):
		return self.eval_grad_

class FEMaterial:
	def hessian(self):
		H = self.get_eval_hessian()
		rows, cols = H.shape

		expr = []
		for i in range(0, rows):
			for j in range(0, cols):
				var = sp.symbols(f'element_matrix[{i*cols + j}*stride]')
				expr.append(ast.Assignment(var, H[i, j]))

		return expr

	def gradient(self):
		g = self.get_eval_gradient()
		rows, cols = g.shape

		expr = []
		for i in range(0, rows):
			var = sp.symbols(f'element_vector[{i}*stride]')
			expr.append(ast.Assignment(var, g[i]))

		return expr

	def value(self):
		var = sp.symbols(f'element_scalar[0]')
		return [ast.Assignment(var, self.get_eval_value())]

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

