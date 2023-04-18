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

#######################################
# Dual basis trafo and weights
#######################################

ls_tet10_trafo_ = [1, 0, 0, 0, sp.Rational(1,5), 0,   sp.Rational(1,5), sp.Rational(1,5), 0,   0,   0, 1, 0, 0, sp.Rational(1,5), sp.Rational(1,5), 0, 0,   sp.Rational(1,5), 0,
                      0, 0, 1, 0, 0,   sp.Rational(1,5), sp.Rational(1,5), 0,   0,   sp.Rational(1,5), 0, 0, 0, 1, 0,   0,   0, sp.Rational(1,5), sp.Rational(1,5), sp.Rational(1,5),
                      0, 0, 0, 0, sp.Rational(3, 5), 0,   0,   0,   0,   0,   0, 0, 0, 0, 0,   sp.Rational(3, 5), 0, 0,   0,   0,
                      0, 0, 0, 0, 0,   0,   sp.Rational(3, 5), 0,   0,   0,   0, 0, 0, 0, 0,   0,   0, sp.Rational(3, 5), 0,   0,
                      0, 0, 0, 0, 0,   0,   0,   0,   sp.Rational(3, 5), 0,   0, 0, 0, 0, 0,   0,   0, 0,   0,   sp.Rational(3, 5) ]


# ls_tet10_weights_ = [7.0,    0.7,    0.7,    0.7,   0.175,  0.7,    0.175,  0.175,  0.7,    0.7,    0.7,  7.0,    0.7,
#                 0.7,    0.175,  0.175,  0.7,   0.7,    0.175,  0.7,    0.7,    0.7,    7.0,    0.7,  0.7,    0.175,
#                 0.175,  0.7,    0.7,    0.175, 0.7,    0.7,    0.7,    7.0,    0.7,    0.7,    0.7,  0.175,  0.175,
#                 0.175,  -3.9,   -3.9,   1.2,   1.2,    5.55,   -1.875, -1.875, -1.875, -1.875, 1.2,  1.2,    -3.9,
#                 -3.9,   1.2,    -1.875, 5.55,  -1.875, 1.2,    -1.875, -1.875, -3.9,   1.2,    -3.9, 1.2,    -1.875,
#                 -1.875, 5.55,   -1.875, 1.2,   -1.875, -3.9,   1.2,    1.2,    -3.9,   -1.875, 1.2,  -1.875, 5.55,
#                 -1.875, -1.875, 1.2,    -3.9,  1.2,    -3.9,   -1.875, -1.875, 1.2,    -1.875, 5.55, -1.875, 1.2,
#                 1.2,    -3.9,   -3.9,   1.2,   -1.875, -1.875, -1.875, -1.875, 5.55 ]

r15div8 = sp.Rational(15, 8)
r7div40 = sp.Rational(7, 40)
r7div10 = sp.Rational(7, 10)
r39div10 = sp.Rational(39, 10)
r6div5 = sp.Rational(6, 5)
r111div20 = sp.Rational(111,20)

ls_tet10_weights_ = [7,    r7div10,    r7div10,    r7div10,   r7div40,  r7div10,    r7div40,  r7div40,  r7div10,    r7div10,    r7div10,  7,    r7div10,
                r7div10,    r7div40,  r7div40,  r7div10,   r7div10,    r7div40,  r7div10,    r7div10,    r7div10,    7,    r7div10,  r7div10,    r7div40,
                r7div40,  r7div10,    r7div10,    r7div40, r7div10,    r7div10,    r7div10,    7,    r7div10,    r7div10,    r7div10,  r7div40,  r7div40,
                r7div40,  -r39div10,   -r39div10,   r6div5,   r6div5,    r111div20,   -r15div8, -r15div8, -r15div8, -r15div8, r6div5,  r6div5,    -r39div10,
                -r39div10,   r6div5,    -r15div8, r111div20,  -r15div8, r6div5,    -r15div8, -r15div8, -r39div10,   r6div5,    -r39div10, r6div5,    -r15div8,
                -r15div8, r111div20,   -r15div8, r6div5,   -r15div8, -r39div10,   r6div5,    r6div5,    -r39div10,   -r15div8, r6div5,  -r15div8, r111div20,
                -r15div8, -r15div8, r6div5,    -r39div10,  r6div5,    -r39div10,   -r15div8, -r15div8, r6div5,    -r15div8, r111div20, -r15div8, r6div5,
                r6div5,    -r39div10,   -r39div10,   r6div5,   -r15div8, -r15div8, -r15div8, -r15div8, r111div20 ]

tet10_trafo = sp.Matrix(10, 10, ls_tet10_trafo_)
tet10_weights = sp.Matrix(10, 10, ls_tet10_weights_)

def tet10_basis_transform(x):
	r, c = tet10_trafo.shape

	ret = sp.Matrix(r, 1, [0]*r)

	for i in range(0, r):
		for j in range(0, c):
			 ret[i] = ret[i] + tet10_trafo[i, j] * x[j]
	return ret

def tet10_basis_transform_expr():
	tx = tet10_basis_transform(coeffs('x', 10))
	ttx = coeffs('values', 10)
	
	expr = []
	
	for i in range(0, 10):
		print(tx[i])
		e = ast.Assignment(ttx[i], tx[i])
		expr.append(e)

	return expr

class WeightedFE(FE):
	def __init__(self, fe, weights, prefix_ = "WeightedFE"):
		self.fe = fe
		self.weights = weights
		self.prefix_  = prefix_

	def fun(self, p):
		fe = self.fe
		weights = self.weights

		nn = fe.n_nodes()
		f = fe.fun(p)
		ret = [0] * nn

		for i in range(0, nn):
			for j in range(0, nn):
				ret[i] += f[j] * weights[i, j]
		return ret

	def name(self):
		return f'{self.prefix_}({self.fe.name()})'
				
	def n_nodes(self):
		return self.fe.n_nodes()

	def manifold_dim(self):
		return self.fe.manifold_dim()

	def spatial_dim(self):
		return self.fe.spatial_dim()

	def integrate(self, q, expr):
		return self.fe.integrate(q, expr)
	
	def jacobian(self, q):
		return self.fe.jacobian(q)

	def jacobian_inverse(self, q):
		return self.fe.jacobian_inverse(q)

	def jacobian_determinant(self, q):
		return self.fe.jacobian_determinant(q)

	def measure(self, q):
		return self.fe.measure(q)

# Factory functions
def TransformedTet10():
	return WeightedFE(Tet10(), tet10_trafo, "Transformed")

def DualTet10():
	return WeightedFE(Tet10(), tet10_weights, "Dual")



