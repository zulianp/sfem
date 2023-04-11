from fe import FE
import sympy as sp

class Tet4(FE):
	def __init__(self):
		super().__init__()

	def name(self):
		return "Tet4"

	def f0(self, x, y, z):
		return 1 - x - y - z

	def f1(self, x, y, z):
		return x

	def f2(self, x, y, z):
		return y

	def f3(self, x, y, z):	
		return z

	def fun(self, p):
		x = p[0]
		y = p[1]
		z = p[2]

		return [
			self.f0(x, y, z),
			self.f1(x, y, z),
			self.f2(x, y, z),
			self.f3(x, y, z)
		]

	def n_nodes(self):
		return 4

	def manifold_dim(self):
		return 3

	def spatial_dim(self):
		return 3

	def integrate(self, q, expr):
		return sp.integrate(expr, (q[2], 0, 1 - q[0] - q[1]), (q[1], 0, 1 - q[0]), (q[0], 0, 1)) 

	def jacobian_inverse(self, q):
		return Ainv

	def measure(self, q):
		return det(A) / 6
