from fe import FE
import sympy as sp
from sfem_codegen import *

class Tri6(FE):
	def __init__(self):
		super().__init__()
		
		self.A_ = sp.Matrix(2, 2, [
			 x1 - x0, x2 - x0,
			 y1 - y0, y2 - y0
			])

		self.Ainv_ = inv2(A)

	def name(self):
		return "Tri6"

	def f0(self, x, y):
	    return 1 - (3 * x) - (3 * y) + (2 * x * x) + (2 * y * y) + (4 * x * y)

	def f1(self, x, y): 
	    return (2 * x * x) - x

	def f2(self, x, y): 
	    return (2 * y * y) - y

	def f3(self, x, y): 
	    return (4 * x) - (4 * x * x) - (4 * x * y)

	def f4(self, x, y): 
	    return (4 * x * y)

	def f5(self, x, y): 
	    return (4 * y) - (4 * x * y) - (4 * y * y)

	def fun(self, p):
		x = p[0]
		y = p[1]

		return [
			self.f0(x, y),
			self.f1(x, y),
			self.f2(x, y),
			self.f3(x, y),
			self.f4(x, y),
			self.f5(x, y)
		]

	def n_nodes(self):
		return 6

	def manifold_dim(self):
		return 2

	def spatial_dim(self):
		return 2

	def integrate(self, q, expr):
		return sp.integrate(expr, (q[1], 0, 1 - q[0]), (q[0], 0, 1)) 

	def jacobian_inverse(self, q):
		return self.Ainv_

	def measure(self, q):
		return det2(self.A_) / 2

class TriShell6(Tri6):
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

	def name(self):
		return "TriShell6"

	def n_nodes(self):
		return 6

	def manifold_dim(self):
		return 2

	def spatial_dim(self):
		return 3

	def jacobian_inverse(self, q):
		return self.Sinv

	def measure(self, q):
		return self.detS / 2

