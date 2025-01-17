#!/usr/bin/env python3

from fe import FE
import sympy as sp
from sfem_codegen import *

class Quad4(FE):
	def __init__(self, isoparam=False):
		super().__init__()
		self.isoparam = isoparam

	def is_isoparametric(self):
		return self.isoparam

	def reference_measure(self):
		return 1

	def barycenter(self):
		return vec2(sp.Rational(1, 2), sp.Rational(1, 2))

	def coords_sub_parametric(self):
		return [[x0, x2], [y0, y2]]

	def coords(self):
		return  [coeffs('px', 4), coeffs('py', 4), coeffs('pz', 4) ]

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
		return self.isoparametric_jacobian(q)

	def jacobian_inverse(self, q):
		return inverse(self.isoparametric_jacobian(q))

	def transform(self, q):
		f = self.fun(q)
		xyz = self.coords()

		pp = sp.zeros(2, 1)

		for i in range(0, 4):
			for d in range(0, self.spatial_dim()):
				pp[d] += xyz[d][i] * f[i]

		return pp

	def inverse_transform(self, p):
		assert False
		return 0

	def jacobian_determinant(self, q):
		return det2(self.jacobian(q))

	def measure(self, q):
		return self.jacobian_determinant(q)

class QuadShell4(Quad4):
	def __init__(self, isoparam=False):
		super().__init__()
		self.isoparam = isoparam

	def is_isoparametric(self):
		return self.isoparam

	def manifold_dim(self):
		return 2

	def spatial_dim(self):
		return 3

	def jacobian(self, q):
		return self.isoparametric_jacobian(q)

	def jacobian_inverse(self, q):
		return pseudo_inverse(self.isoparametric_jacobian(q))

	def coords(self):
		return  [coeffs('px', 4), coeffs('py', 4), coeffs('pz', 4) ]

	def transform(self, q):
		f = self.fun(q)
		xyz = self.coords()

		pp = sp.zeros(3, 1)

		for i in range(0, 4):
			for d in range(0, 3):
				pp[d] += xyz[d][i] * f[i]


		return pp

	def inverse_transform(self, p):
		assert False
		return 0

	def jacobian_determinant(self, q):
		return pseudo_determinant(self.jacobian(q))

	def measure(self, q):
		return self.jacobian_determinant(q)

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
	# AxisAlignedQuad4().generate_c_code()
	QuadShell4(True).generate_qp_based_code()
	