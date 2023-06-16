#!/usr/bin/env python3

from tet4 import *

class BuubbleBase(FE):
	def __init__(self, base):
		super().__init__()
		self.base = base

	def reference_measure(self):
		return self.base.reference_measure()

	def coords_sub_parametric(self):
		return self.base.coords_sub_parametric()

	def n_nodes(self):
		return 1

	def manifold_dim(self):
		return self.base.manifold_dim()

	def spatial_dim(self):
		return 2

	def integrate(self, q, expr):
		return self.base.integrate(q, expr)

	def jacobian(self, q):
		return self.base.jacobian(q)

	def jacobian_inverse(self, q):
		return self.base.jacobian_inverse(q)

	def jacobian_determinant(self, q):
		return self.base.jacobian_determinant(q)

	def measure(self, q):
		return self.base.measure(q)


class Bubble2D(BuubbleBase):
	def __init__(self):
		super().__init__(Tri3())

	def name(self):
		return "Bubble2D"

	def fun(self, p):
		x = p[0]
		y = p[1]

		return [
			self.f0(x, y)
		]

	def f0(self, x, y):
		return 27 * (1 - x - y) * x * y

	
class Bubble3D(BuubbleBase):
	def __init__(self):
		super().__init__(Tet4())

	def name(self):
		return "Bubble3D"

	def fun(self, p):
		x = p[0]
		y = p[1]
		z = p[2]

		return [
			self.f0(x, y, z)
		]

	def f0(self, x, y, z):
		return 256 * (1 - x - y - z) * x * y * z

if __name__ == '__main__':
	Bubble3D().generate_c_code()
