#!/usr/bin/env python3

from sfem_codegen import *
from tet4 import *
from tet10 import *
from tri3 import *
from tri6 import *
from mini import *
from fe_material import *

class StokesOp:
	def __init__(self, fe_u, fe_p):
		fe_velocity = FEFunction("u", fe_u, fe_u.manifold_dim())
		fe_pressure = FEFunction("p", fe_p, 1)
		self.fe_velocity = fe_velocity
		self.fe_pressure = fe_pressure
		
		u_shape_grad = fe_velocity.shape_grad()
		p_shape_grad = fe_pressure.shape_grad()

		nu = len(u_shape_grad)

		mu = sp.symbols('mu')
		self.params = [mu]

		qp = fe_velocity.quadrature_point()
		self.a = sp.Matrix(nu, nu, [0]*(nu*nu))

		for i in range(0,  nu):
			for j in range(0,  nu):
				integr = sp.simplify(mu * fe_u.integrate(qp, mu * inner(u_shape_grad[i], u_shape_grad[j]))) * fe_u.jacobian_determinant(qp)
				self.a[i,j] = integr

		self.mini_condensed_hessian()

	def mini_condensed_hessian(self):
		fe = self.fe_velocity.fe()
		a = self.a
		rows, cols = a.shape
		d = fe.spatial_dim()
		nn = fe.n_nodes()

		C 	= sp.Matrix(d, d, [0]*(d*d))
		B 	= sp.Matrix(d, cols-d, [0]*(d*(cols-d)))
		B_t = sp.Matrix(cols-d, d, [0]*(d*(cols-d)))

		for d1 in range(0, d):
			i1 = d1 * nn + fe.bubble_dof_idx()
			for d2 in range(0, d):
				i2 = d2 * nn + fe.bubble_dof_idx()
				C[d1, d2] = a[i1, i2]

		if d==3:
			invC = inv3(C)
		else:
			assert d == 2
			invC = inv2(C)

		qp = self.fe_velocity.quadrature_point()
		expr = []
		for i in range(0,  d):
			for j in range(0,  d):
				var = sp.symbols(f'element_matrix[{i*d + j}]')
				value = invC[i, j]
				value = subsmat(value, fe.symbol_jacobian_inverse(), fe.jacobian_inverse(qp))
				expr.append(ast.Assignment(var, value))
		c_code(expr)

	def full_hessian(self):
		expr = []
		for i in range(0,  nu):
			for j in range(0,  nu):
				var = sp.symbols(f'element_matrix[{i*nu + j}]')
				value = self.a[i, j]
				value = subsmat(value, fe_u.symbol_jacobian_inverse(), fe_u.jacobian_inverse(qp))
				expr.append(ast.Assignment(var, value))
		c_code(expr)
		return expr


	def gradient_v(self):
		print('// StokesOp::gradient_v')

		u = self.u
		p = self.p

		ufe = u.fe
		pfe = p.fe

	def gradient_q(self):
		print('// StokesOp::gradient_q')

	def apply_hessian_uv(self):
		print('// StokesOp::apply_hessian_uv')

	def hessian_uv(self):
		print('// StokesOp::hessian_uv')

		# Tensor product is done in the C code!
		return self.lapl.hessian()

	def hessian_uq(self):
		print('// StokesOp::hessian_uq')
		return self.div.hessian()

	def hessian_pv(self):
		print('// StokesOp::hessian_pv')

	# Zero block for pq

	class Preconditioner:
		def __init__(self, u, p, qp):
			self.u = u
			self.p = p

			self.lapl = LaplaceOp(u.fe, qp)
			self.mass = MassOp(p, p.test, qp)

		def hessian_uv(self):
			print('// Preconditioner::hessian_uv')
			self.lapl.matrix()

		def hessian_pq(self):
			print('// Preconditioner::hessian_pq')
			return self.mass.matrix()

	def preconditioner(self):
		return Preconditioner(self.u, self.p, self.qp)

def main():
	V = Mini3D()
	Q = Tet4()
	
	# V = Mini2D()
	# Q = Tri3()

	op = StokesOp(V, Q)

	# c_code(op.hessian_uv())
	# c_code(op.hessian_uq())

if __name__ == "__main__":
	main()
