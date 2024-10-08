#!/usr/bin/env python3

from fe import FE
from sfem_codegen import *
from weighted_fe import *

class AAHex8(FE):
	def __init__(self):
		super().__init__()

	def reference_measure(self):
		return 1

	def subparam_n_nodes(self):
		return 2

	def coords_sub_parametric(self):
		return self.coords()

	def coords(self):
		return  [coeffs('x', 2), coeffs('y', 2), coeffs('z', 2) ]


	def fun(self, p):
		x = p[0]
		y = p[1]
		z = p[2]

		f = sp.zeros(8)

		xm = (1 - x)
		ym = (1 - y)
		zm = (1 - z)

		f[0] = xm * ym * zm # (0, 0, 0)
		f[1] = x * ym * zm  # (1, 0, 0)
		f[2] = x * y * zm   # (1, 1, 0)
		f[3] = xm * y * zm  # (0, 1, 0)
		f[4] = xm * ym * z  # (0, 0, 1)
		f[5] = x * ym * z   # (1, 0, 1)
		f[6] = x * y * z    # (1, 1, 1)
		f[7] = xm * y * z   # (0, 1, 1)
		
		return f

	def n_nodes(self):
		return 8

	def manifold_dim(self):
		return 3

	def spatial_dim(self):
		return 3

	def integrate(self, q, expr):
		return sp.integrate(expr, (q[2], 0, 1), (q[1], 0, 1), (q[0], 0, 1)) 
	
	def jacobian(self, q):
		J = sp.zeros(3, 3)

		coords = self.coords()

		for d in range(0, 3):
			J[d, d] = coords[d][1] -  coords[d][0]
			
		return J

	def jacobian_inverse(self, q):
		return inverse(self.jacobian(q))

	def transform(self, q):
		assert False
		return 0

	def inverse_transform(self, p):
		assert False
		return 0

	def jacobian_determinant(self, q):
		return det3(self.jacobian(q))

	def measure(self, q):
		return self.jacobian_determinant(q)