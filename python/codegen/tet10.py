from fe import FE
from sfem_codegen import *

class Tet10(FE):
	def __init__(self):
		super().__init__()

	def name(self):
		return "Tet10"

	def f0(self, x, y, z):
		l0 = 1 - x - y - z
		return (2 * l0 - 1) * l0

	def f1(self, x, y, z):
	    l1 = x
	    return (2 * l1 - 1) * l1

	def f2(self, x, y, z):
	    l2 = y
	    return (2 * l2 - 1) * l2

	def f3(self, x, y, z):
	    l3 = z
	    return (2 * l3 - 1) * l3

	def f4(self, x, y, z):
	    l0 = 1 - x - y - z
	    l1 = x
	    return 4 * l0 * l1

	def f5(self, x, y, z):
	    l1 = x
	    l2 = y
	    return 4 * l1 * l2

	def f6(self, x, y, z):
	    l0 = 1 - x - y - z
	    l2 = y
	    return 4 * l0 * l2

	def f7(self, x, y, z):
	    l0 = 1 - x - y - z
	    l3 = z
	    return 4 * l0 * l3

	def f8(self, x, y, z):
	    l1 = x
	    l3 = z
	    return 4 * l1 * l3

	def f9(self, x, y, z):
	    l2 = y
	    l3 = z
	    return 4 * l2 * l3

	def fun(self, p):
		x = p[0]
		y = p[1]
		z = p[2]

		return [
			self.f0(x, y, z),
			self.f1(x, y, z),
			self.f2(x, y, z),
			self.f3(x, y, z),
			self.f4(x, y, z),
			self.f5(x, y, z),
			self.f6(x, y, z),
			self.f7(x, y, z),
			self.f8(x, y, z),
			self.f9(x, y, z)
		]

	def n_nodes(self):
		return 10

	def manifold_dim(self):
		return 3

	def spatial_dim(self):
		return 3

	def integrate(self, q, expr):
		return sp.integrate(expr, (q[2], 0, 1 - q[0] - q[1]), (q[1], 0, 1 - q[0]), (q[0], 0, 1)) 
	
	def jacobian(self, q):
		return A

	def jacobian_inverse(self, q):
		return Ainv

	def jacobian_determinant(self, q):
		return det3(A)

	def measure(self, q):
		return det3(A) / 6

