#!/usr/bin/env python3

from sfem_codegen import *
from tet4 import *
from tet10 import *
from tri3 import *
from tri6 import *
from mini import *
from fe_material import *

import pdb

# simplify_expr = False
simplify_expr = True

class StokesMiniOp:
	def __init__(self, fe_mini):
		self.fe_mini = fe_mini

		fe_p1 = fe_mini.p1

		if fe_mini.spatial_dim() == 2:
			qp = [qx, qy]
		else:
			assert fe_mini.spatial_dim() == 3
			qp = [qx, qy, qz]

		qp = sp.Matrix(fe_mini.spatial_dim(), 1, qp)
		self.qp = qp
		
		grad_mini = fe_mini.physical_tgrad(qp)
		fun_mini = fe_mini.tfun(qp)
		fun_p1 = fe_p1.fun(qp)

		dV = fe_mini.jacobian_determinant(qp)

		n_vel = len(grad_mini) 
		n_pressure = len(fun_p1)

		mu, rho = sp.symbols('mu rho')
		self.params = [mu, rho]

		self.form_vv = sp.zeros(n_vel, n_vel)
		for i in range(0,  n_vel):
			for j in range(0,  n_vel):
				integr = mu * fe_mini.integrate(qp, inner(grad_mini[i], grad_mini[j])) * dV
				self.form_vv[i, j] = integr

		self.form_vp = sp.zeros(n_vel, n_pressure)
		for i in range(0,  n_vel):
			for j in range(0,  n_pressure):
				integr = -fe_p1.integrate(qp, tr(grad_mini[i]) * fun_p1[j]) * dV
				self.form_vp[i, j] = integr

		self.mass_mini = sp.zeros(n_vel, n_vel)
		self.mass_p1 = sp.zeros(n_pressure, n_pressure)

		bubble_dofs = self.get_bubble_dofs()

		for i in range(0,  n_vel):
			# if i in bubble_dofs:
			# 	continue
			for j in range(0,  n_vel):
				# if j in bubble_dofs:
				# 	continue

				integr = fe_mini.integrate(qp, inner(fun_mini[i], fun_mini[j])) * dV
				self.mass_mini[i, j] = integr

		for i in range(0,  n_pressure):
			for j in range(0,  n_pressure):
				integr = fe_p1.integrate(qp, (fun_p1[i] * fun_p1[j])) * dV
				self.mass_p1[i, j] = integr

		c_u_rhs = coeffs('u_rhs', len(fun_mini))

		# Eliminate bubble coeffcients
		for b in bubble_dofs:
			c_u_rhs[b] = 0

		self.u_rhs = rho * self.mass_mini * c_u_rhs
		self.p_rhs = rho * self.mass_p1 * coeffs('p_rhs', len(fun_p1))

		M_vv, M_vp, M_pv, M_pp, P_v, P_p = self.get_mini_condensed_hessian()
		rhs_v, rhs_p = self.get_mini_condensed_rhs(P_v, P_p)

		v_rows, __ = M_vv.shape
		p_rows, __ = M_pp.shape

		print('---------------')
		print('HESSIAN')
		print('---------------')

		M = sp.zeros(v_rows+p_rows,v_rows+p_rows)
		M[0:v_rows, 0:v_rows] = M_vv[:,:]
		M[0:v_rows, v_rows:(v_rows+p_rows)] = M_vp[:,:]
		M[v_rows:(v_rows+p_rows), 0:v_rows] = M_pv[:,:]
		M[v_rows:(v_rows+p_rows), v_rows:(v_rows+p_rows)] = M_pp[:,:]
		
		c_code(self.assign_matrix(M))
		
		print('---------------')
		print('RHS')
		print('---------------')
	
		rhs = sp.zeros(v_rows + p_rows, 1)
		rhs[0:v_rows,:] = rhs_v[:,:]
		rhs[v_rows:(v_rows+p_rows),:] = rhs_p[:,:]
		c_code(self.assign_vector(rhs))

		self.hessian = M
		self.increment = coeffs('increment', v_rows+p_rows)

		print('---------------')
		print('Apply')
		print('---------------')
		c_code(self.apply())


	# def hessian(self):
	def apply(self):
		H = self.hessian
		rows, cols = H.shape
		x = self.increment

		Hx = H * x
		return self.assign_vector(Hx)

	def gradient(self):
		return self.apply()

	def get_bubble_dofs(self):
		fe_mini = self.fe_mini
		d = fe_mini.spatial_dim()
		nn_mini = fe_mini.n_nodes()

		bubble_dofs = []
		for d1 in range(0, d):
			ii = d1 * nn_mini + fe_mini.bubble_dof_idx()
			bubble_dofs.append(ii)
		return bubble_dofs

	def form_vv_matrix(self):
		fe_mini = self.fe_mini
		fe_p1 = fe_mini.p1
		fe_bubble = fe_mini.bubble
		qp = self.qp
		d = fe_mini.spatial_dim()
		nn_mini = fe_mini.n_nodes()
		nn_p1 = fe_p1.n_nodes()
		nn_bubble = fe_bubble.n_nodes()

		A = sp.zeros(nn_p1   * d,   nn_p1 * d)
		C = sp.zeros(nn_bubble * d, nn_bubble * d)
		B = sp.zeros(nn_bubble * d, nn_p1 * d)

		bubble_dofs = self.get_bubble_dofs()
		for d1 in range(0, len(bubble_dofs)):
			i1 = bubble_dofs[d1]
			for d2 in range(0, len(bubble_dofs)):
				i2 = bubble_dofs[d2]
				C[d1, d2] = self.form_vv[i1, i2]

		for d1 in range(0, len(bubble_dofs)):
			i1 = bubble_dofs[d1]
			idx = 0
			for i2 in range(0, nn_mini*d):
				if i2 in bubble_dofs:
					continue
				B[d1, idx] = self.form_vv[i1, i2]
				idx += 1

		ii = 0
		for i in range(0, nn_mini*d):
			if i in bubble_dofs:
				continue

			jj = 0
			for j in range(0, nn_mini*d):
				if j in bubble_dofs:
					continue

				A[ii, jj] = self.form_vv[i, j]
				jj += 1
			ii += 1
		return A, B, C

	def form_vp_matrix(self):
		fe_mini = self.fe_mini
		fe_p1 = fe_mini.p1
		fe_bubble = fe_mini.bubble
		qp = self.qp
		d = fe_mini.spatial_dim()
		nn_mini = fe_mini.n_nodes()
		nn_p1 = fe_p1.n_nodes()
		nn_bubble = fe_bubble.n_nodes()

		D = sp.zeros(nn_p1*d, nn_p1)
		E = sp.zeros(nn_bubble*d ,nn_p1)

		bubble_dofs = self.get_bubble_dofs()

		ii = 0
		for d1 in range(0, nn_mini*d):
			if d1 in bubble_dofs:
				continue

			for i2 in range(0, nn_p1):
				D[ii, i2] = self.form_vp[d1, i2]
			ii += 1

		for d1 in range(0, len(bubble_dofs)):
			i1 = bubble_dofs[d1]
			for i2 in range(0, nn_p1):
				E[d1, i2] = self.form_vp[i1, i2]
		return D, E

	def get_mini_condensed_hessian(self):
		A, B, C = self.form_vv_matrix()
		D, E = self.form_vp_matrix()

		# print(B)
		# print(E)
		# print(C)


		d = self.fe_mini.spatial_dim()

		if d == 3:
			C_inv = inv3(C)
		else:
			assert d == 2
			C_inv = inv2(C)

		M_vv = A - B.T * C_inv * B
		M_vp = -(B.T * C_inv * E) + D;
		M_pv = (D.T - E.T * C_inv * B)
		M_pp = -E.T * C_inv * E

		P_v = -B.T * C_inv
		P_p = -E.T * C_inv
		return M_vv, M_vp, M_pv, M_pp, P_v, P_p

	def get_mini_condensed_rhs(self, P_v, P_p):
		u_rhs = self.u_rhs
		p_rhs = self.p_rhs

		bubble_dofs = self.get_bubble_dofs()

		nbubble = len(bubble_dofs)
		bubble_rhs = sp.zeros(nbubble, 1)

		for i in range(0, nbubble):
			bubble_rhs[i] = u_rhs[bubble_dofs[i]]

		ndofs = len(u_rhs)
		cn = ndofs - nbubble
		p1_u_rhs = sp.zeros(cn, 1)

		ii = 0
		for i in range(0, ndofs):
			if i in bubble_dofs:
				continue
			p1_u_rhs[ii] = u_rhs[i]
			ii += 1

		rhs_v = p1_u_rhs + P_v * bubble_rhs
		rhs_p = p_rhs + P_p * bubble_rhs
		return rhs_v, rhs_p

	def assign_tensor(self, name, a):
		fe = self.fe_mini
		qp = self.qp
		rows, cols = a.shape

		expr = []
		for i in range(0,  rows):
			for j in range(0,  cols):
				var = sp.symbols(f'{name}[{i*cols + j}]')
				value = a[i, j]

				value = subsmat(value, fe.symbol_jacobian_inverse(), fe.jacobian_inverse(qp))

				if simplify_expr:
					value = sp.simplify(value)
					
				expr.append(ast.Assignment(var, value))
		return expr

	def assign_matrix(self, a):
		return self.assign_tensor("element_matrix", a)

	def assign_vector(self, a):
		return self.assign_tensor("element_vector", a)

def main():
	op = StokesMiniOp(Mini2D())
	# op = StokesMiniOp(Mini3D())

if __name__ == "__main__":
	main()
