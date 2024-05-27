#!/usr/bin/env python3

from fe import FE
import sympy as sp
from sfem_codegen import *
from weighted_fe import  *
from lagrange_triangle import *

class Tri10(FE):
	def __init__(self):
		super().__init__()
		
		self.A_ = sp.Matrix(2, 2, [
			 x1 - x0, x2 - x0,
			 y1 - y0, y2 - y0
			])

		self.Ainv_ = inv2(self.A_)

	def subparam_n_nodes(self):
		return 3

	def coords_sub_parametric(self):
		return [[x0, x1, x2], [y0, y1, y2]]

	def name(self):
		return "Tri10"

	def fun(self, p):
		x = p[0]
		y = p[1]

		f = sp.Matrix(10, 1, lagrange_triangle(3, x, y))
		return f
 		
	def n_nodes(self):
		return 10

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

	def reference_measure(self):
		return sp.Rational(1, 2)

class TriShell10(Tri10):
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

	def coords_sub_parametric(self):
		return [[x0, x1, x2], [y0, y1, y2], [z0, z1, z2]]

	def name(self):
		return "TriShell10"

	def n_nodes(self):
		return 10

	def manifold_dim(self):
		return 2

	def spatial_dim(self):
		return 3

	def jacobian_inverse(self, q):
		return self.Sinv

	def jacobian_determinant(self, q):
		return self.detS 

	def measure(self, q):
		return self.detS / 2

	def reference_measure(self):
		return sp.Rational(1, 2)

if __name__ == '__main__':
	Tri10().generate_c_code()
	TriShell10().generate_c_code()
