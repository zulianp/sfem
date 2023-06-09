#!/usr/bin/env python3

from sfem_codegen import *

class FESymbol:
	def __init__(self, var, expansion):
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

		if ncomp == 1:
			self.symbols_value_ = sp.symbols(f's_{name}')
		else:
			self.symbols_value_ = coeffs(f's_{name}', ncomp)
		self.symbols_grad_ = sp.Matrix(fe.spatial_dim(), ncomp, coeffs(f's_grad_{name}', fe.spatial_dim()*ncomp))
		
		# Basis functions and grads
		self.symbols_trial_fun_ = coeffs(f's_trial_fun_{name}', ncomp)
		self.symbols_test_fun_  = coeffs(f's_test_fun_{name}', ncomp)

		self.symbols_trial_grad_ = sp.Matrix(fe.spatial_dim(), 1, coeffs(f's_trial_grad_{name}', fe.spatial_dim()))
		self.symbols_test_grad_  = sp.Matrix(fe.spatial_dim(), 1, coeffs(f's_test_grad_{name}', fe.spatial_dim()))

		q = [qx, qy, qz]
		q = sp.Matrix(fe.manifold_dim(), 1, q[0:fe.manifold_dim()])

		self.q_ = q
		self.shape_fun_  = fe.tfun(q, ncomp)
		self.shape_grad_ = fe.physical_tgrad(q, ncomp)
		
		# Evaluation
		val = self.coeffs_[0] * self.shape_fun_[0]
		for i in range(1, len(self.shape_fun_)):
			val += self.coeffs_[i] * self.shape_fun_[i]

		self.eval_value_ = val

		val = self.coeffs_[0]  *self.shape_grad_[0]
		for i in range(1, len(self.shape_grad_)):
			val += self.coeffs_[i]  *self.shape_grad_[i]

		self.eval_grad_ = val

	def name(self):
		return self.name_

	def coeffs(self):
		return self.coeffs_

	def value(self):
		return FESymbol(self.symbols_value_, self.eval_value_)

	def grad(self):
		return FESymbol(self.symbols_grad_, self.eval_grad_)

	def trial_fun(self):
		return FESymbol(self.symbols_trial_fun_, self.shape_fun_)

	def test_fun(self):
		return FESymbol(self.symbols_test_fun_, self.shape_fun_)  

	def trial_grad(self):
		return FESymbol(self.symbols_trial_grad_, self.shape_grad_)

	def test_grad(self):
		return FESymbol(self.symbols_test_grad_, self.shape_grad_)  

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
	# def matrix_derivative(self, expr, x):
	# 	rows, cols = x.shape
	# 	ret = sp.Matrix(rows, cols, [0]*(rows*cols))
	# 	for r in range(0, rows):
	# 		for c in range(0, cols):
	# 			ret[r, c] = sp.diff(expr, x[r, c])
	# 	return ret

	# def vector_derivative(self, expr, x):
	# 	rows = len(x)
	# 	ret = sp.Matrix(rows, 1, [0]*(rows))
	# 	for r in range(0, rows):
	# 		ret[r] = sp.diff(expr, x[r])
	# 	return ret

	# def matrix_directional_derivative(self, expr, x, h):
	# 	B = self.matrix_derivative(expr, x)
	# 	return inner(B, h)

	# def vector_directional_derivative(self, expr, x, h):
	# 	b = self.vector_derivative(expr, x)
	# 	return inner(b, h)

	def derivative(self, expr, x):
		rows, cols = x.shape
		ret = sp.Matrix(rows, cols, [0]*(rows*cols))
		for r in range(0, rows):
			for c in range(0, cols):
				ret[r, c] = sp.diff(expr, x[r, c])
		return ret

	def directional_derivative(self, expr, x, h):
		B = self.derivative(expr, x)
		return inner(B, h)

	def inner_list(self, expr, list_of_funs):
		ret = []

		print(len(list_of_funs))
		for t in list_of_funs:
			ret.append(inner(expr, t))
		return sp.Matrix(len(ret), 1, ret)

	def mult_list(self, expr, list_of_funs):
		ret = []
		for t in list_of_funs:
			ret.append(expr * t)
		return sp.Matrix(len(ret), 1, ret)

	def vector_diff_scalar_x(self, vector_expr, x, h):
		rows = len(h)
		cols = len(vector_expr)

		ret = sp.Matrix(rows, cols, [0]*(rows*cols))

		for j in range(0, cols):
			djdx = sp.diff(vector_expr[j], x)

			for i in range(0, rows):
				ret[i, j] = djdx * h[i]
		return ret

	def vector_diff_tensor_x(self, vector_expr, x, h):
		rows = len(h)
		cols = len(vector_expr)

		dim1, dim2 = x.shape
		dfdx = sp.Matrix(dim1, dim2, [0]*(dim1*dim2))
		ret = sp.Matrix(rows, cols, [0]*(rows*cols))

		for j in range(0, cols):
			for d1 in range(0, dim1):
				for d2 in range(0, dim2):
					dfdx[d1,d2] = sp.diff(vector_expr[j], x[d1,d2])

			for i in range(0, rows):
				ret[i, j] = inner(dfdx, h[i])
		return ret


	def integrate(self, fe, q, mat):
		rows, cols = mat.shape
		ret = sp.Matrix(rows, cols, [0]*(rows*cols))

		for i in range(0, rows):
			for j in range(0, cols):
				ret[i, j] = fe.integrate(q, mat[i, j])
		return ret

	def subs_tensors(self, pairs, mat):
		rows, cols = mat.shape
		ret = sp.Matrix(rows, cols, [0]*(rows*cols))

		for i in range(0, rows):
			for j in range(0, cols):
				ret[i, j] = mat[i, j]
				for p in pairs:
					k, v = p
					# print('-----------------')
					# print(f'subs | {k[0,0]} | with | {v[0,0]} |')
					# print('-----------------')
					ret[i, j] = subsmat(ret[i, j], k, v)
		return ret

	def scale_tensors(self, scale_factor, mat):
		rows, cols = mat.shape
		ret = sp.Matrix(rows, cols, [0]*(rows*cols))

		for i in range(0, rows):
			for j in range(0, cols):
				ret[i, j] = mat[i, j] * scale_factor
		return ret

	def subs(self, pairs, value):
		ret = value
		for p in pairs:
			k, v = p
			ret = subsmat(ret, k, v)
		return ret

	def assign_matrix(self, mat):
		rows, cols = mat.shape

		expr = []
		for i in range(0, rows):
			for j in range(0, cols):
				var = sp.symbols(f'element_matrix[{i*cols + j}]')
				expr.append(ast.Assignment(var, mat[i, j]))
		return expr

	def assign_vector(self, vec):
		rows, cols = vec.shape

		expr = []
		for i in range(0, rows):
			var = sp.symbols(f'element_vector[{i}]')
			expr.append(ast.Assignment(var, vec[i]))
		return expr

	def hessian(self):
		H = self.get_eval_hessian()
		return self.assign_matrix(H)

	def gradient(self):
		g = self.get_eval_gradient()
		return self.assign_vector(g)

	def value(self):
		var = sp.symbols(f'element_scalar[0]')
		return [ast.Assignment(var, self.get_eval_value())]

	def apply(self):
		H = self.integr_hessian
		rows, cols = H.shape
		self.increment = coeffs('increment', cols)
		inc = self.increment

		expr = []
		for i in range(0, rows):
			val = 0
			for j in range(0, cols):
				val += H[i, j] * inc[j]

			var = sp.symbols(f'element_vector[{i}]')
			expr.append(ast.Assignment(var, val))
		return expr

