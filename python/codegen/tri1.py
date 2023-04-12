from fe import FE
import sympy as sp
from sfem_codegen import *

class Tri1(FE):
	def __init__(self):
		super().__init__()

		self.A_ = sp.Matrix(2, 2, [
			 x1 - x0, x2 - x0,
			 y1 - y0, y2 - y0
			])

		self.Ainv_ = inv2(A)

	def name(self):
		return "Tri1"

	def f0(self, x, y):
		return 1

	def fun(self, p):
		x = p[0]
		y = p[1]

		return [
			self.f0(x, y)
		]

	def n_nodes(self):
		return 1

	def manifold_dim(self):
		return 2

	def spatial_dim(self):
		return 2

	def integrate(self, q, expr):
		return sp.integrate(expr, (q[1], 0, 1 - q[0]), (q[0], 0, 1)) 

	def jacobian(self, q):
		return self.A_

	def jacobian_inverse(self, q):
		return self.Ainv_

	def jacobian_determinant(self, q):
		return det2(self.A_)

	def measure(self, q):
		return det2(self.A_) / 2

class TriShell1(Tri1):
	def __init__(self):
		super().__init__()

		self.S = sp.Matrix(3, 2, [
			 x1 - x0, x2 - x0,
			 y1 - y0, y2 - y0,
			 z1 - z0, z2 - z0
			])

		StS = self.S.T * self.S
		self.detS = sp.sqrt(det2(StS))
		self.Sinv = inv2(StS) * self.S.T
	
	def name(self):
		return "TriShell1"

	def manifold_dim(self):
		return 2

	def spatial_dim(self):
		return 3

	def jacobian_inverse(self, q):
		return self.Sinv

	def measure(self, q):
		return self.detS / 2

