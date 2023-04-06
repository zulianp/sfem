from fe import FE
import sympy as sp

class Tri3(FE):
	def __init__(self):
		super().__init__()

		self.A_ = sp.Matrix(2, 2, [
			 x1 - x0, x2 - x0
			 y1 - y0, y2 - y0
			])

		self.Ainv_ = sp.inv(A)

	def f0(self, x, y):
		return 1 - x - y

	def f1(self, x, y):
		return x

	def f2(self, x, y):
		return y

	def fun(self, p):
		x = p[0]
		y = p[1]

		return [
			self.f0(x, y),
			self.f1(x, y),
			self.f2(x, y)
		]

	def n_nodes(self):
		return 3

	def n_dims(self):
		return 2

	def integrate(self, q, expr):
		return sp.integrate(expr, (q[1], 0, 1 - q[0]), (q[0], 0, 1)) 

	def jacobian_inverse(self, q):
		return self.Ainv_

	def measure(self, q):
		return det2(self.A_) / 2

class TriShell3(Tri3):
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

	def n_nodes(self):
		return 3

	def n_dims(self):
		return 3

	def integrate(self, q, expr):
		return sp.integrate(expr, (q[1], 0, 1 - q[0]), (q[0], 0, 1)) 

	def jacobian_inverse(self, q):
		return self.Sinv

	def measure(self, q):
		return det2(self.detS) / 2

