#!/usr/bin/env python3

from sfem_codegen import *
import matplotlib.pyplot as plt

# https://nalu-wind.readthedocs.io/en/latest/source/theory/discretizationApproach.html

# Quad4 nodes
p0 = sp.Matrix(2, 1, [x0, y0])
p1 = sp.Matrix(2, 1, [x1, y1])
p2 = sp.Matrix(2, 1, [x2, y2])

# Integration points
bary  = sp.Matrix(2, 1, [sp.Rational(1, 3), sp.Rational(1, 3)])
m0 = sp.Matrix(2, 1, [sp.Rational(1, 2), 0])
m1 = sp.Matrix(2, 1, [sp.Rational(1, 2), sp.Rational(1, 2)])
m2 = sp.Matrix(2, 1, [0, sp.Rational(1, 2)])

ip0 = (bary + m0)/2
ip1 = (bary + m1)/2
ip2 = (bary + m2)/2

qx = [ip0[0], ip1[0], ip2[0]]
qy = [ip0[1], ip1[1], ip2[1]]

def perp(e):
 	return sp.Matrix(2, 1, [-e[1], e[0]])

def fun(x, y):
	return [ 1 - x - y, x, y ]

def p(x, y):
	f = fun(x, y)
	px = f[0] * p0[0] + f[1] * p1[0] + f[2] * p2[0]
	py = f[0] * p0[1] + f[1] * p1[1] + f[2] * p2[1]
	return [px, py]

def assign_matrix(name, mat):
	rows, cols = mat.shape
	expr = []
	for i in range(0, rows):
		for j in range(0, cols):
			var = sp.symbols(f'{name}[{i*cols + j}]')
			expr.append(ast.Assignment(var, mat[i, j]))
	return expr

def cv_interp(v):
	vx = sp.zeros(3, 1)
	for i in range(0, 3):
		f = fun(qx[i], qy[i])
		for j in range(0, 3):
			vx[i] += f[j] * v[j]
	return vx

def cv_normals():
	b = (p0 + p1 + p2)/3
	dn = []
	for i in range(0, 3):
		pi = p(qx[i], qy[i])
		ei = [ pi[0] - b[0], pi[1] - b[1]]
		dn.append(2 * perp(ei))
	return dn

def advective_fluxes(vx, vy, dn):
	q = []
	for i in range(0, 3):
		qi = vx[i] * dn[i][0] + vy[i] * dn[i][1]
		q.append(qi)

	return q

def pw_max(a, b):
	return sp.Piecewise((b, a < b), (a, True))

def advection_op(q):
	A = sp.zeros(3, 3)

	# Node 0
	A[0, 0] = -pw_max( q[0], 0) - pw_max(-q[2], 0)
	A[0, 1] =  pw_max(-q[0], 0)
	A[0, 2] =  pw_max(q[2], 0)

	# Node 1
	A[1, 1] = -pw_max(-q[0], 0) - pw_max(q[1], 0)
	A[1, 0] =  pw_max(q[0], 0)
	A[1, 2] =  pw_max(-q[1], 0)

	# Node 2
	A[2, 2] = -pw_max(-q[1], 0) - pw_max(q[2], 0)
	A[2, 0] = pw_max(-q[2], 0)
	A[2, 1] = pw_max(q[1], 0)
	return A

def ref_subs(expr):
	expr = expr.subs(x0, 0)
	expr = expr.subs(y0, 0)

	expr = expr.subs(x1, 1)
	expr = expr.subs(y1, 0)
	
	expr = expr.subs(x2, 0)
	expr = expr.subs(y2, 1)
	return expr

vx = coeffs('vx', 3)
vy = coeffs('vy', 3)

vcx = cv_interp(vx)
vcy = cv_interp(vy)

dn = cv_normals()
q = advective_fluxes(vcx, vcy, dn)



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

# Check on ref element
if False:
# if True:
	print('------------------')
	print(qx)
	print(qy)
	print('------------------')
	for dni in dn:
		print(f'({ref_subs(dni[0])}, {ref_subs(dni[1])})')

	print('------------------')
	for i in range(0, 3):

		line = ""

		for j in range(0, 3):
			su = ref_subs(A[i, j])

			for v in vx:
				su = su.subs(v, 0)

			for v in vy:
				su = su.subs(v, 1)

			line += f"{round(su, 1)} "

		print(line)
		print('\n')
