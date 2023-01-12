#!/usr/bin/env python3

import numpy as np
import sympy as sp
from sympy.utilities.codegen import codegen
import sympy.codegen.ast as ast

def n_test_functions():
	return 4*3

def det3(mat):
    return mat[0, 0] * mat[1, 1] * mat[2, 2] + mat[0, 1] * mat[1, 2] * mat[2, 0] + mat[0, 2] * mat[1, 0] * mat[2, 1] - mat[0, 0] * mat[1, 2] * mat[2, 1] - mat[0, 1] * mat[1, 0] * mat[2, 2] - mat[0, 2] * mat[1, 1] * mat[2, 0]

def inv3(mat):
	# Sympy version (same but slower)
	# return mat.inv()
    mat_inv = sp.zeros(3, 3)
    det = det3(mat)
    mat_inv[0, 0] = (mat[1, 1] * mat[2, 2] - mat[1, 2] * mat[2, 1]) / det
    mat_inv[0, 1] = (mat[0, 2] * mat[2, 1] - mat[0, 1] * mat[2, 2]) / det
    mat_inv[0, 2] = (mat[0, 1] * mat[1, 2] - mat[0, 2] * mat[1, 1]) / det
    mat_inv[1, 0] = (mat[1, 2] * mat[2, 0] - mat[1, 0] * mat[2, 2]) / det
    mat_inv[1, 1] = (mat[0, 0] * mat[2, 2] - mat[0, 2] * mat[2, 0]) / det
    mat_inv[1, 2] = (mat[0, 2] * mat[1, 0] - mat[0, 0] * mat[1, 2]) / det
    mat_inv[2, 0] = (mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0]) / det
    mat_inv[2, 1] = (mat[0, 1] * mat[2, 0] - mat[0, 0] * mat[2, 1]) / det
    mat_inv[2, 2] = (mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]) / det
    return mat_inv

def c_code(expr):
	sub_expr, simpl_expr = sp.cse(expr)
	printer = sp.printing.c.C99CodePrinter()
	lines = []

	for var,expr in sub_expr:
	    lines.append(f'real_t {var} = {printer.doprint(expr)};')

	for v in simpl_expr:
	        lines.append(printer.doprint(v))

	code_string=f'\n'.join(lines)

	print(code_string)

# if __name__ == '__main__':
mu, lmbda = sp.symbols('mu lambda')

# Element coordinates
x0, x1, x2, x3 = sp.symbols('x0 x1 x2 x3')
y0, y1, y2, y3 = sp.symbols('y0 y1 y2 y3')
z0, z1, z2, z3 = sp.symbols('z0 z1 z2 z3')

qx, qy, qz = sp.symbols('qx qy qz')

# Displacement
u0, u1, u2 = sp.symbols('u0 u1 u2')

u = sp.Matrix(3, 1, [u0, u1, u2])

# Displacement gradient
du0dx, du0dy, du0dz = sp.symbols('du0dx du0dy d0udz')
du1dx, du1dy, du1dz = sp.symbols('du1dx du1dy d1udz')
du2dx, du2dy, du2dz = sp.symbols('du2dx du2dy d2udz')

gradu = sp.Matrix(3, 3, [
	du0dx, du0dy, du0dz, 
	du1dx, du1dy, du1dz, 
	du2dx, du2dy, du2dz]
)

# Quadrature points (Physical coordinates)
q = sp.Matrix(3, 1, [qx, qy, qz])

# Affine transformation
A = sp.Matrix(3, 3, [
	 x1 - x0, x2 - x0, x3 - x0,
	 y1 - y0, y2 - y0, y3 - y0,
	 z1 - z0, z2 - z0, z3 - z0,
	])

Ainv = inv3(A)

b = sp.Matrix(3, 1, [x0, y0, z0])

def ref_fun(x, y, z):
	return [
	 1 - x - y - z, 
	 x,
	 y,
	 z
	]

def fun(x, y, z):
	xmb = x - b[0]
	ymb = y - b[1]
	zmb = z - b[2]

	xref = Ainv[0, 0] * xmb + Ainv[0, 1] * ymb  + Ainv[0, 2] * zmb
	yref = Ainv[1, 0] * xmb + Ainv[1, 1] * ymb  + Ainv[1, 2] * zmb
	zref = Ainv[2, 0] * xmb + Ainv[2, 1] * ymb  + Ainv[2, 2] * zmb
	return ref_fun(xref, yref, zref)


def tgrad(x, y, z):
	ret = []
	f = fun(x, y, z)

	i = 0
	for fi in f:
		gix = sp.simplify(sp.diff(fi, x))
		giy = sp.simplify(sp.diff(fi, y))
		giz = sp.simplify(sp.diff(fi, z))
		g = [gix, giy, giz]

		for d1 in range(0, 3):
			eps = sp.Matrix(3, 3, [0, 0, 0, 
								   0, 0, 0, 
								   0, 0, 0])

			for d2 in range(0, 3):
				eps[d1, d2] = g[d2]

			simmetrized_eps = (eps + eps.T) / 2
			ret.append(simmetrized_eps)

		i += 1
	return ret

def linear_strain(gradu):
	return (gradu + gradu.T) / 2

def inner(l, r):
	ret = 0
	for d1 in range(0, 3):
		for d2 in range(0, 3):
			ret += l[d1, d2] * r[d1, d2]

	return ret

def tr(mat):
	ret = 0
	for d1 in range(0, 3):
		ret += mat[d1, d1]
	return ret

dV = det3(A) / 6
eps = tgrad(qx, qy, qz)

# Elastic energy
epsu = linear_strain(gradu)

e = lmbda/2 * tr(epsu) * tr(epsu) + mu * inner(epsu, epsu)

def makeenergy():
	integr = sp.simplify(e * det3(A))
	integr = sp.integrate(integr, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1))
	sintegr = sp.simplify(integr)

	form = sp.symbols(f'element_energy')
	energy_expr = (ast.Assignment(form, sintegr))	
	return energy_expr

# Gradient
dedu = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0])

for d1 in range(0, 3):
	for d2 in range(0, 3):
		dedu[d1, d2] = sp.diff(e, gradu[d1, d2])
		# print(f'{d1}, {d2}) {dedu[d1, d2]}')

grade = [0]*(4*3)

for i in range(0, 4*3):
	integr =  inner(dedu, eps[i])
	grade[i] = integr

def makegrad(i, q):
	integr =  inner(dedu, eps[i])
	grade[i] = integr

	# Simplify expressions (switch comment on lines for reducing times)
	integr = sp.integrate(integr * det3(A), (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1))
	# sintegr = sp.simplify(integr)
	sintegr = integr
	lform = sp.symbols(f'element_vector[{i}]')
	expr = ast.Assignment(lform, sintegr)
	q.put(expr)

def print_grads():
	for i in range(0, 4*3):
		print(f'{i}) {grad_expr[i]}')

# Hessian
def makehessian(i, q):
	tuples = []

	He = sp.Matrix(3, 3, 
		[0, 0, 0, 
		 0, 0, 0,
		 0, 0, 0])


	for d1 in range(0, 3):
		for d2 in range(0, 3):
			He[d1, d2] = sp.diff(grade[i], gradu[d1, d2])

	for j in range(i, 4*3):
		# Bilinear form
		integr = inner(He, eps[j]) * dV
		
		# Simplify expressions (switch comment on lines for reducing times)
		sintegr = sp.simplify(integr)
		# sintegr = integr

		# Store results in array
		bform1 = sp.symbols(f'element_matrix[{i*(4*3)+j}]')

		tuples.append((i, j, ast.Assignment(bform1, sintegr)))

		# Take advantage of symmetry to reduce code-gen times
		if i != j:
			bform2 = sp.symbols(f'element_matrix[{i+(4*3)*j}]')
			tuples.append((j, i, ast.Assignment(bform2, sintegr)))

	q.put(tuples)

