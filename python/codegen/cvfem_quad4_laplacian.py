#!/usr/bin/env python3

from sfem_codegen import *
import matplotlib.pyplot as plt

# https://nalu-wind.readthedocs.io/en/latest/source/theory/discretizationApproach.html

# Quad4 nodes
p0 = sp.Matrix(2, 1, [x0, y0])
p1 = sp.Matrix(2, 1, [x1, y1])
p2 = sp.Matrix(2, 1, [x2, y2])
p3 = sp.Matrix(2, 1, [x3, y3])

# Integration points
num_qx = [0.5,  0.75, 0.5,  0.25]
num_qy = [0.25, 0.5,  0.75, 0.5 ]

def perp(e):
 	return sp.Matrix(2, 1, [-e[1], e[0]])

def fun(x, y):
	return [ (1 - x) * (1 - y), x * (1 - y), x * y, (1 - x) * y ]

def p(x, y):
	f = fun(x, y)
	px = f[0] * p0[0] + f[1] * p1[0] + f[2] * p2[0] + f[3] * p3[0];
	py = f[0] * p0[1] + f[1] * p1[1] + f[2] * p2[1] + f[3] * p3[1]
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
	vx = sp.zeros(4, 1)
	for i in range(0, 4):
		f = fun(num_qx[i], num_qy[i])
		for j in range(0, 4):
			vx[i] += f[j] * v[j]
	return vx

def cv_normals():
	b = (p0 + p1 + p2 + p3)/4
	dn = []
	for i in range(0, 4):
		pi = p(num_qx[i], num_qy[i])
		ei = [ pi[0] - b[0], pi[1] - b[1]]
		dn.append(2 * perp(ei))
	return dn

def laplace_op(dn):
	A = sp.zeros(4, 4)

	shape = fun(qx, qy)
	grad_shape = []

	for i in range(0, 4):
		grad_shape.append(sp.Matrix(2, 1, [
			sp.diff(shape[i], qx),
			sp.diff(shape[i], qy)
		]))

	dnlut = [
		[3, 0],
		[0, 1],
		[1, 2],
		[2, 3],
	]

	dnsign = [
		[1, -1],
		[1, -1],
		[1, -1],
		[1, -1]
	]

	for i in range(0, len(num_qx)):
		lut_i = dnlut[i]
		sign_i = dnsign[i]

		for l in range(0, len(lut_i)):
			k = lut_i[l]
			s = sign_i[l]
			dn_l = sp.simplify(s * dn[k])

			for j in range(0, 4):
				gx = grad_shape[j][0].subs(qx, num_qx[k]).subs(qy, num_qy[k])
				gy = grad_shape[j][1].subs(qx, num_qx[k]).subs(qy, num_qy[k])
				A[i, j] += dn_l[0] * gx + dn_l[1] * gy
	# A = -A
	return A

def ref_subs(expr):
	expr = expr.subs(x0, 0)
	expr = expr.subs(y0, 0)

	expr = expr.subs(x1, 1)
	expr = expr.subs(y1, 0)
	
	expr = expr.subs(x2, 1)
	expr = expr.subs(y2, 1)

	expr = expr.subs(x3, 0)
	expr = expr.subs(y3, 1)
	return expr

dn = cv_normals()

print('----------------------------')
print('Hessian')
print('----------------------------')

A = laplace_op(dn)
expr = assign_matrix('element_matrix', A)
c_code(expr)

print('----------------------------')
print('Apply')
print('----------------------------')

x = coeffs('x', 4)
y = A * x
expr = assign_matrix('element_vector', y)
c_code(expr)


# # Check on ref element
if True:
	for i in range(0, 4):

		line = ""

		for j in range(0, 4):
			su = ref_subs(A[i, j])
			line += f"{round(su, 1)} "

		print(line)
		print('\n')
