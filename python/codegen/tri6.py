from fe import FE
import sympy as sp
from sfem_codegen import *
from weighted_fe import  *

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

	def jacobian(self, q):
		return self.A_

	def jacobian_inverse(self, q):
		return self.Ainv_

	def jacobian_determinant(self, q):
		return det2(self.A_)

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

	def jacobian_determinant(self, q):
		return self.detS 

	def measure(self, q):
		return self.detS / 2

# Hard-coded alpha 1./5.
r24div5 = sp.Rational(24, 5)
r4div5 = sp.Rational(4, 5)
r1div5 = sp.Rational(1, 5)
r33div10 = sp.Rational(33, 10)
r6div5 = sp.Rational(6, 5)
r69div20 = sp.Rational(69, 20)
r57div40 = sp.Rational(57, 40)

ls_tri6_weights_ = [r24div5, r4div5,  r4div5, -r1div5, r4div5, -r1div5, r4div5, r24div5, r4div5, -r1div5, -r1div5, r4div5,
              r4div5, r4div5,  r24div5,  r4div5, -r1div5, -r1div5,   -r33div10, -r33div10, r6div5,  r69div20,   -r57div40, -r57div40,
              r6div5, -r33div10, -r33div10, -r57div40, r69div20, -r57div40, -r33div10, r6div5,  -r33div10, -r57div40, -r57div40, r69div20 ]

alpha = sp.Rational(1, 5)
ls_tri6_trafo_ = [0] * (6*6)

ls_tri6_trafo_[0] = 1
ls_tri6_trafo_[3] = alpha
ls_tri6_trafo_[5] = alpha

ls_tri6_trafo_[7] = 1
ls_tri6_trafo_[9] = alpha
ls_tri6_trafo_[10] = alpha

ls_tri6_trafo_[14] = 1
ls_tri6_trafo_[16] = alpha
ls_tri6_trafo_[17] = alpha

ls_tri6_trafo_[21] = (1 - 2 * alpha)
ls_tri6_trafo_[28] = (1 - 2 * alpha)
ls_tri6_trafo_[35] = (1 - 2 * alpha)

tri6_trafo = sp.Matrix(6, 6, ls_tri6_trafo_)
tri6_weights = sp.Matrix(6, 6, ls_tri6_weights_)

#######################################
# Dual basis trafo and weights
#######################################

# Factory functions
def TransformedTri6():
	return WeightedFE(Tri6(), tri6_trafo, "Transformed")

def DualTri6():
	return WeightedFE(Tri6(), tri6_weights, "Dual")

def TransformedTriShell6():
	return WeightedFE(TriShell6(), tri6_trafo, "Transformed")

def DualTriShell6():
	return WeightedFE(TriShell6(), tri6_weights, "Dual")


def tri6_basis_transform(x):
	r, c = tri6_trafo.shape

	ret = sp.Matrix(r, 1, [0]*r)

	trafo = tri6_trafo.T

	for i in range(0, r):
		for j in range(0, c):
			 ret[i] = ret[i] + trafo[i, j] * x[j]
	return ret

def tri6_basis_transform_expr():
	tx = tri6_basis_transform(coeffs('x', 6))
	ttx = coeffs('values', 6)
	
	expr = []
	
	for i in range(0, 6):
		e = ast.Assignment(ttx[i], tx[i])
		expr.append(e)

	return expr


