#!/usr/bin/env python3

from fe import FE
import sympy as sp
from sfem_codegen import *

class AxisAlignedQuad4(FE):
	def __init__(self):
		super().__init__()

		self.A_ = sp.Matrix(2, 2, [
			 x2 - x0, 0,
			 0, y2 - y0
			])

		self.Ainv_ = inv2(self.A_)
		# print(self.Ainv_)

	def reference_measure(self):
		return 1

	def symbol_jacobian_inverse(self):
		# Remove off-diags for generator efficiency
		J = super().symbol_jacobian_inverse()
		J[0, 1]  = 0
		J[1, 0] = 0
		return J

	def coords_sub_parametric(self):
		return [[x0, x2], [y0, y2]]

	def name(self):
		return "Quad4"

	def f0(self, x, y):
		return (1 - x) * (1 - y)

	def f1(self, x, y):
		return  x * (1 - y)

	def f2(self, x, y):
		return x * y

	def f3(self, x, y):
		return (1 - x)  * y

	def fun(self, p):
		x = p[0]
		y = p[1]

		return [
			self.f0(x, y),
			self.f1(x, y),
			self.f2(x, y),
			self.f3(x, y)
		]

	def n_nodes(self):
		return 4

	def manifold_dim(self):
		return 2

	def spatial_dim(self):
		return 2

	def integrate(self, q, expr):
		return sp.integrate(expr, (q[1], 0, 1), (q[0], 0, 1)) 

	def jacobian(self, q):
		return self.A_

	def jacobian_inverse(self, q):
		return self.Ainv_

	def jacobian_determinant(self, q):
		return det2(self.A_)

	def measure(self, q):
		return det2(self.A_)


if __name__ == '__main__':
	AxisAlignedQuad4().generate_c_code()