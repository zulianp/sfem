#!/usr/bin/env python3

from sfem_codegen import *
from quad4 import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *
from tet20 import *
from symbolic_fe import *

from time import perf_counter

def simplify(expr):
	return expr;
	# return sp.simplify(expr)

def assign_matrix(name, mat):
	rows, cols = mat.shape
	expr = []
	for i in range(0, rows):
		for j in range(0, cols):
			var = sp.symbols(f'{name}[{i*cols + j}]')
			expr.append(ast.Assignment(var, mat[i, j]))
	return expr

class HyperElasticity:
	def __init__(self, fe, model):
		fe.use_adjugate = True

		F = model.deformation_gradient_symb()
		Psi = model.energy()
		dims = model.dims
		q = q_point(dims)

		# FE gradients
		trial_grad 	   = matrix_coeff('trial_grad', dims, dims)
		test_grad  	   = matrix_coeff('test_grad', dims, dims)
		inc_grad_symb  = matrix_coeff('inc_grad', dims, dims)
		disp_grad_symb = matrix_coeff('disp_grad', dims, dims)
		displacement   = coeffs('u', dims * fe.n_nodes())
		increment 	   = coeffs('h', dims * fe.n_nodes())
		value 	   = coeffs('v', dims * fe.n_nodes())

		ref_grad = fe.tgrad(q)
		jac_inv = fe.symbol_jacobian_inverse_as_adjugate()

		disp_grad = sp.zeros(dims, dims)
		for i in range(0, fe.n_nodes() * fe.manifold_dim()):
			disp_grad += displacement[i] * ref_grad[i]
		disp_grad = disp_grad * jac_inv

		inc_grad = sp.zeros(dims, dims)
		for i in range(0, fe.n_nodes() * fe.manifold_dim()):
			inc_grad += increment[i] * ref_grad[i]
		inc_grad = inc_grad * jac_inv

		vec_grad = sp.zeros(dims, dims)
		for i in range(0, fe.n_nodes() * fe.manifold_dim()):
			vec_grad += value[i] * ref_grad[i]
		vec_grad = vec_grad * jac_inv

		# First Piola
		P = sp.zeros(dims, dims)
		for d1 in range(0, dims):
			for d2 in range(0, dims):
				P[d1, d2] = simplify(sp.diff(Psi, F[d1, d2]))

		# let us switch test with trial (exploit symmetry)
		gj = inner(P, trial_grad)

		# Stress linearization
		lin_stress = sp.zeros(dims, dims)
		for d1 in range(0, dims):
			for d2 in range(0, dims):
				lin_stress[d1, d2] =  simplify(sp.diff(gj, F[d1, d2]))

		######################################
		# Store common quantities
		######################################
		self.fe = fe
		self.q = q
		self.jac_inv = jac_inv
		self.dV = fe.reference_measure() * fe.symbol_jacobian_determinant() * fe.quadrature_weight()

		self.model = model

		self.trial_grad 	= trial_grad
		self.test_grad  	= test_grad
		self.inc_grad_symb  = inc_grad_symb
		self.disp_grad_symb = disp_grad_symb
		self.F = F

		self.dims = dims
		self.Psi = Psi
		self.P = P
		self.lin_stress = lin_stress
		self.loperand_symb = matrix_coeff('loperand', dims, dims)

		self.disp_grad = disp_grad
		self.inc_grad = inc_grad
		self.vec_grad = vec_grad


	def apply(self):
		lin_stress = self.lin_stress
		trial_grad = self.trial_grad
		test_grad  = self.test_grad
		inc_grad_symb = self.inc_grad_symb 
		jac_inv = self.jac_inv
		dV = self.dV
		dims = self.fe.spatial_dim()

		loperand = lin_stress.T * (jac_inv.T * dV)

		expr = loperand.copy()
		for d1 in range(0, dims):
			for d2 in range(0, dims):
				expr[d1, d2] = subsmat(expr[d1, d2], trial_grad, inc_grad_symb)


		# Hij = simplify(inner(loperand, trial_grad))
		# expr = subsmat(Hij, test_grad, inc_grad_symb)
		# c_code(assign_matrix('loperand', expr))

		# c_code(assign_matrix('lin_stress', lin_stress))


		lform = []
		for d1 in range(0, dims):
			val = 0
			for d2 in range(0, dims):
				val += self.loperand_symb[d1, d2] * test_grad[0, d2]

			lform.append(ast.AddAugmentedAssignment(sp.symbols(f'lform[{d1}]'), val))

		buffers = []
		# buffers.extend(assign_matrix('F', self.disp_grad_symb + sp.eye(dims, dims)))
		buffers.extend(assign_matrix('vec_grad', self.vec_grad))
		ret = {
			'buffers': buffers,
			'F' : assign_matrix('F', self.disp_grad_symb + sp.eye(dims, dims)),
			 'loperand' : assign_matrix('loperand', expr),
			'lform'   : lform
		}

		return ret

class HyperElasticModel:
	def __init__(self, dims):
		self.dims = dims
		self.J_symb  	  = sp.symbols('J')
		self.trC_symb     = sp.symbols('trC')
		self.F_symb  	  = matrix_coeff('F', dims, dims)
		self.F_inv_t_symb = matrix_coeff('F_inv_t', dims, dims)
		self.F_inv_symb   = matrix_coeff('F_inv', dims, dims)
		self.C_symb       = matrix_coeff('C', dims, dims)

		self.J 		 = determinant(self.F_symb)
		self.F_inv   = inverse(self.F_symb)
		self.F_inv_t = self.F_inv.T
		self.C 		 = self.F_symb.T * self.F_symb
		self.trC     = sp.trace(self.C)


class NeoHookeanOgden(HyperElasticModel):
	def __init__(self, dims):
		super().__init__(dims)
		self.name = 'NeoHookeanOgden'
		mu, lmbda = sp.symbols('mu lambda')
		self.params = [(mu, 1.0), (lmbda, 1.0)]

		trC = self.trC
		J 	= self.J
		
		self.fun = mu/2 *(trC - dims) - mu * sp.log(J) + (lmbda/2) * (sp.log(J))**2
		
	def deformation_gradient_symb(self):
		return self.F_symb

	def energy(self):
		return self.fun


def main():
	start = perf_counter()

	fe = Tet10()
	model = NeoHookeanOgden(fe.spatial_dim())
	op = HyperElasticity(fe, model)

	op_apply = op.apply()
	for k, v in op_apply.items():
		print('-------------------------------')
		print(f'{k}')
		print('-------------------------------')
		c_code(v)
		print('-------------------------------')

	stop = perf_counter()
	console.print(f'Overall: {stop - start} seconds')


if __name__ == '__main__':
	main()