#!/usr/bin/env python3

from fe import FE
from sfem_codegen import *

class Tet4(FE):
	def __init__(self):
		super().__init__()

	def reference_measure(self):
		return sp.Rational(1, 6)

	def coords_sub_parametric(self):
		return [[x0, x1, x2, x3], [y0, y1, y2, y3], [z0, z1, z2, z3]]

	def barycenter(self):
		return vec3(sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 4))

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

	def jacobian(self, q):
		return A

	def jacobian_inverse(self, q):
		return Ainv

	def jacobian_determinant(self, q):
		return det3(A)

	def measure(self, q):
		return det3(A) / 6

	def transform(self, q):
		return self.jacobian(q) * q + sp.Matrix(3, 1, [x0, y0, z0])

	def inverse_transform(self, p):
		diff = (p - sp.Matrix(3, 1, [x0, y0, z0]))
		return self.jacobian_inverse(p) * diff

if __name__ == '__main__':
	# Tet4().generate_c_code()

	# c_code(Tet4().measure(vec3(qx, qy, qz)))
	Tet4().generate_qp_based_code()