#!/usr/bin/env python3

from sfem_codegen import *
from tet4 import *
from tet10 import *
from tri3 import *
from tri6 import *
from mini import *
from fe_material import *

import pdb

# A * u + BT * ub  - f = 0
# B * u + C * ub - fb = 0
# ub = Cinv * (fb - B * u)
# A * u + BT * Cinv * (fb - B * u) - f = 0

# statically condensed system:
# (A - BT * Cinv * B) * u = f - BT * Cinv * fb
# (A - S) * u = f - P * fb


class StokesOp:
	def __init__(self, fe_u, fe_p):
		fe_velocity = FEFunction("u", fe_u, fe_u.manifold_dim())
		fe_pressure = FEFunction("p", fe_p, 1)
		self.fe_velocity = fe_velocity
		self.fe_pressure = fe_pressure
		
		u_shape_grad = fe_velocity.shape_grad()
		u_shape_fun = fe_velocity.shape_fun()
		p_shape_fun = fe_pressure.shape_fun()

		nu = len(u_shape_grad)
		np = len(p_shape_fun)

		mu = sp.symbols('mu')
		self.params = [mu]

		qp = fe_velocity.quadrature_point()
		# Assume affine
		dV = fe_u.jacobian_determinant(qp)

		self.a = sp.Matrix(nu, nu, [0]*(nu*nu))
		for i in range(0,  nu):
			for j in range(0,  nu):
				integr = mu * fe_u.integrate(qp, inner(u_shape_grad[i], u_shape_grad[j])) * dV
				# integr = sp.simplify(integr)
				self.a[i, j] = integr

		self.b = sp.Matrix(nu, np, [0] * (nu*np))
		for i in range(0,  nu):
			for j in range(0,  np):
				integr = -fe_u.integrate(qp, tr(u_shape_grad[i]) * p_shape_fun[j]) * dV
				# integr = sp.simplify(integr)
				self.b[i, j] = integr


		self.u_mass_mat = sp.Matrix(nu, nu, [0] * (nu * nu))
		self.p_mass_mat = sp.Matrix(np, np, [0] * (np * np))

		for i in range(0,  nu):
			for j in range(0,  nu):
				integr = fe_u.integrate(qp, inner(u_shape_fun[i], u_shape_fun[j])) * dV
				# integr = sp.simplify(integr)
				self.u_mass_mat[i, j] = integr

		for i in range(0,  np):
			for j in range(0,  np):
				integr = fe_u.integrate(qp, (p_shape_fun[i] * p_shape_fun[j])) * dV
				# integr = sp.simplify(integr)
				self.p_mass_mat[i, j] = integr

		self.u_rhs = self.u_mass_mat * coeffs('u_rhs', len(u_shape_fun))
		self.p_rhs = self.p_mass_mat * coeffs('p_rhs', len(p_shape_fun))
		
		print('---------------')
		print('HESSIAN')
		print('---------------')
		Hc, P, bubble_dofs = self.get_mini_condensed_hessian()
		c_code(self.assign_matrix(Hc))
		# self.hessian_uu()
		# self.hessian_up()
		# self.hessian()
		# self.project_bubble_rhs()

		print('---------------')
		print('RHS')
		print('---------------')
		self.get_mini_condensed_rhs()
		# self.apply()

	def get_mini_condensed_hessian(self):
		u_fe = self.fe_velocity.fe()
		p_fe = self.fe_pressure.fe()
		qp = self.fe_velocity.quadrature_point()
		d = u_fe.spatial_dim()
		nn = u_fe.n_nodes()

		H = self.get_hessian()
		rows, cols = H.shape

		p1_rows = cols-d

		A = sp.Matrix(p1_rows, p1_rows, [0]*((p1_rows)*(p1_rows)))
		C = sp.Matrix(d, d, [0]*(d*d))
		B = sp.Matrix(d, p1_rows, [0]*(d*(p1_rows)))

		for d1 in range(0, d):
			i1 = d1 * nn + u_fe.bubble_dof_idx()
			for d2 in range(0, d):
				i2 = d2 * nn + u_fe.bubble_dof_idx()
				C[d1, d2] = H[i1, i2]

		for d1 in range(0, d):
			i1 = d1 * nn + u_fe.bubble_dof_idx()
			idx = 0
			for d2 in range(0, d):
				for j in range(0, nn):
					if j == u_fe.bubble_dof_idx():
						continue
					j2 = d2 * nn + j
					B[d1, idx] = H[i1, j2]
					idx += 1

			for j in range(nn*d, cols):
				B[d1, idx] = H[i1, j]
				idx += 1

		bubble_dofs = []
		for d1 in range(0, d):
			i1 = d1 * nn + u_fe.bubble_dof_idx()
			bubble_dofs.append(i1)

		ii = 0
		for i in range(0, rows):
			if i in bubble_dofs:
				continue

			jj = 0
			for j in range(0, cols):
				if j in bubble_dofs:
					continue

				A[ii, jj] = H[i, j]
				jj += 1
			ii += 1

		if d == 3:
			C_inv = inv3(C)
		else:
			assert d == 2
			C_inv = inv2(C)

		# c_code(self.assign_matrix(C_inv))
		# c_code(self.assign_matrix(B))
		# c_code(self.assign_matrix(A))

		S = B.T * C_inv * B
		P = B.T * C_inv

		Hc = A - S

		print(S.shape)
		print(C_inv.shape)
		print(B.shape)
		
		# c_code(self.assign_matrix(P))
		return Hc, P, bubble_dofs

	def get_mini_condensed_rhs(self):
		rhs = self.get_rhs()
		__, P, bubble_dofs = self.get_mini_condensed_hessian()

		nbubble = len(bubble_dofs)
		bubble_rhs = sp.Matrix(nbubble, 1, [0] *nbubble)

		for i in range(0, nbubble):
			bubble_rhs[i] = rhs[bubble_dofs[i]]

		ndofs = len(rhs)
		cn = ndofs - nbubble
		crhs = sp.Matrix(cn, 1, [0] * (cn))

		ii = 0
		for i in range(0, ndofs):
			if i in bubble_dofs:
				continue
			crhs[ii] = rhs[i]
			ii += 1

		ret = crhs - P * bubble_rhs

		# c_code(self.assign_vector(rhs))
		c_code(self.assign_vector(ret))
		return ret

	def assign_tensor(self, name, a):
		fe = self.fe_velocity.fe()
		qp = self.fe_velocity.quadrature_point()
		rows, cols = a.shape

		expr = []
		for i in range(0,  rows):
			for j in range(0,  cols):
				var = sp.symbols(f'{name}[{i*cols + j}]')
				value = a[i, j]
				value = subsmat(value, fe.symbol_jacobian_inverse(), fe.jacobian_inverse(qp))
				expr.append(ast.Assignment(var, value))
		return expr

	def assign_matrix(self, a):
		return self.assign_tensor("element_matrix", a)

	def assign_vector(self, a):
		return self.assign_tensor("element_vector", a)

	def hessian_uu(self):
		expr = self.assign_matrix(self.a)
		c_code(expr)
		return expr

	def hessian_up(self):
		expr = self.assign_matrix(self.b)
		c_code(expr)
		return expr

	def hessian(self):
		H = self.get_hessian()
		expr = self.assign_matrix(H)
		c_code(expr)
		return expr

	def get_hessian(self):
		fe_velocity = self.fe_velocity
		fe_pressure = self.fe_pressure
		dims = fe_velocity.fe().manifold_dim()
		ndofs = fe_velocity.fe().n_nodes() * dims + fe_pressure.fe().n_nodes()
		H = sp.Matrix(ndofs, ndofs, [0]*(ndofs*ndofs))

		a = self.a
		b = self.b 

		arows,acols = a.shape
		brows,bcols = b.shape
		
		H[0:arows, 0:arows] = a
		H[0:arows, arows:ndofs] = b
		H[arows:ndofs, 0:arows] = b.T
		return H

	def get_rhs(self):
		fe_velocity = self.fe_velocity
		fe_pressure = self.fe_pressure
		dims = fe_velocity.fe().manifold_dim()
		ndofs = fe_velocity.fe().n_nodes() * dims + fe_pressure.fe().n_nodes()
		g = sp.Matrix(ndofs, 1, [0]*(ndofs))

		u_rhs = self.u_rhs 
		p_rhs = self.p_rhs 

		arows,__ = u_rhs.shape

		# pdb.set_trace()
		g[0:arows,:] = u_rhs[:]
		g[arows:ndofs,:] = p_rhs[:]
		return g


	def project_bubble_rhs(self):
		Hc, P = self.get_mini_condensed_hessian()
		u_fe = self.fe_velocity.fe()

		fb = coeffs('rhs_bubble', u_fe.manifold_dim())

		Pfb = P * fb
		c_code(self.assign_vector(Pfb))
		return Pfb

	def apply(self):
		Hc, P = self.get_mini_condensed_hessian()
		rows, cols = Hc.shape
		x = coeffs('x', cols)

		Hx = Hc * x
		c_code(self.assign_vector(Hx))
		return Hx

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
	# class Preconditioner:
	# 	def __init__(self, u, p, qp):
	# 		self.u = u
	# 		self.p = p

	# 		self.lapl = LaplaceOp(u.fe, qp)
	# 		self.mass = MassOp(p, p.test, qp)

	# 	def hessian_uv(self):
	# 		print('// Preconditioner::hessian_uv')
	# 		self.lapl.matrix()

	# 	def hessian_pq(self):
	# 		print('// Preconditioner::hessian_pq')
	# 		return self.mass.matrix()

	# def preconditioner(self):
	# 	return Preconditioner(self.u, self.p, self.qp)

def main():
	if False:
		V = Mini3D()
		Q = Tet4()
	else:
		V = Mini2D()
		Q = Tri3()

	op = StokesOp(V, Q)

	# c_code(op.hessian_uv())
	# c_code(op.hessian_uq())

if __name__ == "__main__":
	main()
