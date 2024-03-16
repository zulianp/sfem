#!/usr/bin/env python3

from sfem_codegen import *
import matplotlib.pyplot as plt

def perp(e):
 	return sp.Matrix(2, 1, [-e[1], e[0]])
 	
def p(x, y):
	f = [1 - x - y, x, y]
	return f[0] * p0 + f[1] * p1 + f[2] * p2

def assign_matrix(name, mat):
	rows, cols = mat.shape

	expr = []
	for i in range(0, rows):
		for j in range(0, cols):
			var = sp.symbols(f'{name}[{i*cols + j}]')
			expr.append(ast.Assignment(var, mat[i, j]))
	return expr


def create_jacobian():
	A2 = sp.Matrix(2, 2, [
		 x1 - x0, x2 - x0,
		 y1 - y0, y2 - y0
	])

	return A2

def cv_interp(v):
	w0 = sp.Rational(5, 12)
	w1 = sp.Rational(1, 6)

	vc0 = (w0 * v[0]) * (w0 * v[1]) * (w1 * v[2])
	vc1 = (w1 * v[0]) * (w0 * v[1]) * (w0 * v[2])
	vc2 = (w0 * v[0]) * (w1 * v[1]) * (w0 * v[2])

	return vc0, vc1, vc2

def cv_normals(J):
	a = sp.Matrix(2, 1, [0., 0.])
	b = J[:, 0]
	c = J[:, 1]

	bary = (a + b + c) / 3
	e1   = (a + b) / 2
	e2   = (b + c) / 2
	e3   = (c + a) / 2

	s1   = e1 - bary
	s2   = e2 - bary
	s3   = e3 - bary

	dn1 = perp(s1)
	dn2 = perp(s2)
	dn3 = perp(s3)
	return dn1, dn2, dn3

def advective_fluxes(vxc, vyc, dn1, dn2, dn3):
	q0 = vxc[0] * dn1[0] + vyc[0] * dn1[1]
	q1 = vxc[1] * dn2[0] + vyc[1] * dn2[1]
	q2 = vxc[2] * dn3[0] + vyc[2] * dn3[1]
	return q0, q1, q2

def pw_max(a, b):
	return sp.Piecewise((b, a < b), (a, True))

def advection_op(q):
	A = sp.zeros(3, 3)

	# Node a
	A[0, 0] = -pw_max( q[0], 0) - pw_max(-q[2], 0)
	A[0, 1] = pw_max(-q[0], 0)
	A[0, 2] = pw_max(q[2], 0)

	# Node b
	A[1, 1] = -pw_max(-q[0], 0) - pw_max(q[1], 0)
	A[1, 0] = pw_max(q[0], 0)
	A[1, 2] = pw_max(-q[1], 0)

	# Node c
	A[2, 2] = -pw_max(-q[1], 0) - pw_max(q[2], 0)
	A[2, 0] = pw_max(-q[2], 0)
	A[2, 1] = pw_max(q[1], 0)

	return A

vx = coeffs('vx', 3)
vy = coeffs('vy', 3)

vcx0, vcx1, vcx2 = cv_interp(vx)
vcy0, vcy1, vcy2 = cv_interp(vy)

vcx = [vcx0, vcx1, vcx2]
vcy = [vcy0, vcy1, vcy2]

J = matrix_coeff('J', 2, 2)
dn0, dn1, dn2 = cv_normals(J)
q0, q1, q2 = advective_fluxes(vcx, vcy, dn0, dn1, dn2)
q = [q0, q1, q2]

print('----------------------------')
print('Jacobian')
print('----------------------------')

JJ = create_jacobian()
JJexpr = assign_matrix('J', JJ)
c_code(JJexpr)

print('----------------------------')
print('Hessian')
print('----------------------------')

A = advection_op(q)
expr = assign_matrix('element_matrix', A)
c_code(expr)

print('----------------------------')
print('Apply')
print('----------------------------')

x = coeffs('x', 3)
y = A * x
expr = assign_matrix('element_vector', y)
c_code(expr)
