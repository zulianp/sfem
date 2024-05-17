#!/usr/bin/env python3

# https://what-when-how.com/the-finite-element-method/fem-for-3d-solids-finite-element-method-part-3/

from fe import FE
from sfem_codegen import *
from weighted_fe import *

class Tet20(FE):
	def __init__(self):
		super().__init__()

	def reference_measure(self):
		return sp.Rational(1, 6)

	def subparam_n_nodes(self):
		return 4

	def coords_sub_parametric(self):
		return [[x0, x1, x2, x3], [y0, y1, y2, y3], [z0, z1, z2, z3]]

	def name(self):
		return "Tet20"

	def fun(self, p):
		x = p[0]
		y = p[1]
		z = p[2]

		l1 = 1 - x - y - z
		l2 = x
		l3 = y
		l4 = z

		f = sp.zeros(20, 1)

		# Corners nodes
		l = [l1, l2, l3, l4]
		for i in range(0, 4):
			f[i] = sp.Rational(1, 2) * (3 * l[i] - 1) * (3 * l[i]  - 2) * l[i]

		# Edge nodes
		f[4] = sp.Rational(9, 2) * (3 * l1 - 1) * l1 * l3
		f[5] = sp.Rational(9, 2) * (3 * l3 - 1) * l1 * l3
		f[6] = sp.Rational(9, 2) * (3 * l1 - 1) * l1 * l2
		f[7] = sp.Rational(9, 2) * (3 * l2 - 1) * l1 * l2
		f[8] = sp.Rational(9, 2) * (3 * l2 - 1) * l2 * l3
		f[9] = sp.Rational(9, 2) * (3 * l3 - 1) * l2 * l3

		f[10] = sp.Rational(9, 2) * (3 * l1 - 1) * l1 * l4
		f[11] = sp.Rational(9, 2) * (3 * l4 - 1) * l1 * l4
		f[12] = sp.Rational(9, 2) * (3 * l2 - 1) * l2 * l4
		f[13] = sp.Rational(9, 2) * (3 * l4 - 1) * l2 * l4
		f[14] = sp.Rational(9, 2) * (3 * l3 - 1) * l3 * l4
		f[15] = sp.Rational(9, 2) * (3 * l4 - 1) * l3 * l4

		# Center surface nodes
		f[16] = 27 * l2 * l3 * l4
		f[17] = 27 * l1 * l2 * l3
		f[18] = 27 * l1 * l3 * l4
		f[19] = 27 * l1 * l2 * l4
		return f

	def n_nodes(self):
		return 20

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

if __name__ == '__main__':
	Tet20().generate_c_code()
