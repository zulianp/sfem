#!/usr/bin/env python3

from fe import FE
from sfem_codegen import *
from weighted_fe import *

class Hex8(FE):
	def __init__(self):
		super().__init__()

	def reference_measure(self):
		return 1

	def subparam_n_nodes(self):
		return 4

	def coords_sub_parametric(self):
		xyz = self.coords()
		x = xyz[0]
		y = xyz[1]
		z = xyz[2]

		return [
			[x[0], x[1], x[3], x[4]],
			[y[0], y[1], y[3], y[4]],
			[z[0], z[1], z[3], z[4]]
		]

	def coords(self):
		return  [coeffs('x', 10), coeffs('y', 10), coeffs('z', 10) ]


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
		return self.isoparametric_jacobian(q)

	def jacobian_inverse(self, q):
		return inverse(self.isoparametric_jacobian(q))

	def transform(self, q):
		f = self.fun(q)
		xyz = self.coords()

		pp = sp.zeros(3, 1)

		for i in range(0, 8):
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


def assign_matrix(name, mat):
	rows, cols = mat.shape
	expr = []
	for i in range(0, rows):
		for j in range(0, cols):
			var = sp.symbols(f'{name}[{i*cols + j}]')
			expr.append(ast.Assignment(var, mat[i, j]))
	return expr

def points_from_sub_ref_hex8():
	x0, x1, y0, y1, z0, z1 = sp.symbols('px0 px1 py0 py1 pz0 pz1')
	hex8 = Hex8()

	qs = [
		[x0, x1, x1, x0, x0, x1, x1, x0],
		[y0, y0, y1, y1, y0, y0, y1, y1],
		[z0, z0, z0, z0, z1, z1, z1, z1]
	]

	x = sp.zeros(8, 1)
	y = sp.zeros(8, 1)
	z = sp.zeros(8, 1)

	for k in range(0, 8):
		q = [qs[0][k], qs[1][k], qs[2][k]]
		f = hex8.fun(q)
		xyz = hex8.coords()

		for i in range(0, 8):
			fi = sp.simplify(f[i])
			x[k] += xyz[0][i] * fi
			y[k] += xyz[1][i] * fi
			z[k] += xyz[2][i] * fi
	
	expr = assign_matrix('lx', x)
	expr.extend(assign_matrix('ly', y))
	expr.extend(assign_matrix('lz', z))
	
	c_code(expr)


def assign_fff(name, mat):
	rows, cols = mat.shape

	expr = []
	idx = 0
	for i in range(0, rows):
		for j in range(i, cols):
			var = sp.symbols(f'{name}[{idx}]')
			expr.append(ast.Assignment(var, mat[i, j]))
			idx += 1
	return expr

def sub_fff():
	c = coeffs("fff", 6)
	FFF = sp.Matrix(3, 3, [
		c[0], c[1], c[2], 
		c[1], c[3], c[4],
		c[2], c[4], c[5]
	])

	h = sp.symbols('h')

	A = sp.Matrix(3, 3, [
		h, 0, 0,
		0, h, 0,
		0, 0, h
	])
	
	Aminv = inv3(A)
	detAm = determinant(A)

	sub_FFF = Aminv * FFF * Aminv.T * detAm

	expr = assign_fff('sub_fff', sub_FFF)
	c_code(expr)

if __name__ == '__main__':
	# Hex8().generate_qp_based_code()
	# points_from_sub_ref_hex8()
	sub_fff()

	