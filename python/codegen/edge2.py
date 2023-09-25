#!/usr/bin/env python3

from fe import FE
import sympy as sp
from sfem_codegen import *

class Edge2(FE):
	def __init__(self):
		super().__init__()

		self.A_ = sp.Matrix(1, 1, [
			 x1 - x0
			])

		self.Ainv_ = 1/self.A_[0]

	def name(self):
		return "Edge2"

	def f0(self, x):
		return 1 - x

	def f1(self, x):
		return x

	def fun(self, p):
		x = p[0]

		return [
			self.f0(x),
			self.f1(x)
		]

	def n_nodes(self):
		return 2

	def manifold_dim(self):
		return 1

	def spatial_dim(self):
		return 1

	def integrate(self, q, expr):
		return sp.integrate(expr, (q[0], 0, 1)) 

	def jacobian(self, q):
		return self.A_

	def jacobian_inverse(self, q):
		return self.Ainv_

	def jacobian_determinant(self, q):
		return det2(self.A_)

	def measure(self, q):
		return det2(self.A_)

class EdgeShell2(Edge2):
	def __init__(self):
		super().__init__()

		self.S = sp.Matrix(2, 1, [
			 x1 - x0,
			 y1 - y0
			])

		StS = self.S.T * self.S
		self.detS = sp.sqrt(StS[0])
		self.Sinv = (1/StS[0])* self.S.T

	def inverse_transform(self, p):
		return self.Sinv * (p - sp.Matrix(2, 1, [x0, y0]))

	def name(self):
		return "Edge2hell2"
		
	def manifold_dim(self):
		return 1

	def spatial_dim(self):
		return 2

	def jacobian_inverse(self, q):
		return self.Sinv

	def jacobian_determinant(self, q):
		return self.detS 

	def measure(self, q):
		return self.detS


if __name__ == '__main__':
	# Tri3().generate_c_code()
	# TriShell3().generate_c_code()

	EdgeShell2().generate_qp_based_code()