#!/usr/bin/env python3

from fe import FE
from sfem_codegen import *
from weighted_fe import *
from lagrange import uniform_lagr

class Hex27(FE):
	grid_ordering=True

	def __init__(self, isoparam=False):
		super().__init__()
		self.isoparam = isoparam

	def is_isoparametric(self):
		return self.isoparam

	def reference_measure(self):
		return 1

	def subparam_n_nodes(self):
		return 2

	def barycenter(self):
		return vec3(sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2))

	def coords_sub_parametric(self):
		xyz = self.coords()
		x = xyz[0]
		y = xyz[1]
		z = xyz[2]

		return [
			[x[0], x[26]],
			[y[0], y[26]],
			[z[0], z[26]]
		]

	def coords(self):
		return  [coeffs('x', 27), coeffs('y', 27), coeffs('z', 27) ]

	def fun(self, p):
		x = p[0]
		y = p[1]
		z = p[2]

		f = sp.zeros(27, 1)

		lx = uniform_lagr(2, x, sp.Rational(1, 2))
		ly = uniform_lagr(2, y, sp.Rational(1, 2))
		lz = uniform_lagr(2, z, sp.Rational(1, 2))

		for k in range(0, 3):
			for j in range(0, 3):
				for i in range(0, 3):
					f[k*9+j*3+i] = lz[k] * ly[j] * lx[i]
		return f

	def n_nodes(self):
		return 27

	def manifold_dim(self):
		return 3

	def spatial_dim(self):
		return 3

	def integrate(self, q, expr):
		return sp.integrate(expr, (q[2], 0, 1), (q[1], 0, 1), (q[0], 0, 1)) 
	
	def jacobian(self, q):
		return self.isoparametric_jacobian(q)

	def jacobian_inverse(self, q):
		return inverse(self.isoparametric_jacobian(q))

	def transform(self, q):
		f = self.fun(q)
		xyz = self.coords()

		pp = sp.zeros(3, 1)

		for i in range(0, 27):
			for d in range(0, 3):
				pp[d] += xyz[d][i] * f[i]

		return pp

	def inverse_transform(self, p):
		assert False
		return 0

	def jacobian_determinant(self, q):
		return det3(self.jacobian(q))

	def measure(self, q):
		return self.jacobian_determinant(q)

if __name__ == '__main__':
	print("TODO")
