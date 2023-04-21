#!/usr/bin/env python3

from sfem_codegen import *
from tet4 import *
from tet10 import *
from fields import *

from laplace_op import LaplaceOp
from div_op import DivOp
# from u_dot_grad_q_op import UDotGradQOp

from mass_op import MassOp

class StokesOp:
	def __init__(self, u, p, qp):
		self.u = u
		self.p = p
		self.qp = qp

		self.lapl = LaplaceOp(u.fe, qp)
		self.div = DivOp(u, p.fe, qp)

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
	V = Tet10()
	Q = Tet4()

	u = VectorField(V, [
		coeffs('ux', V.n_nodes()),
		coeffs('uy', V.n_nodes()),
		coeffs('uz', V.n_nodes()),
	])

	p = Field(Q, coeffs('p', Q.n_nodes()))
	op = StokesOp(u, p, [qx, qy, qz])

	# c_code(op.hessian_uv())
	c_code(op.hessian_uq())

if __name__ == "__main__":
	main()
