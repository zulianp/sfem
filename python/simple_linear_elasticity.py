#!/usr/bin/env python3

import numpy as np
import sympy as sp
from sympy.utilities.codegen import codegen
import sympy.codegen.ast as ast

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

mu, lmbda = sp.symbols('mu lambda')

# Element coordinates
x0, x1, x2, x3 = sp.symbols('x0 x1 x2 x3')
y0, y1, y2, y3 = sp.symbols('y0 y1 y2 y3')
z0, z1, z2, z3 = sp.symbols('z0 z1 z2 z3')
qx, qy, qz = sp.symbols('qx qy qz')

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

# Preallocate (for preserving write order)
expr = [0]*(4*3*4*3)

for i in range(0, 4*3):
	for j in range(i, 4*3):
		# Bilinear form
		integr = (2 * mu * inner(eps[i], eps[j]) + lmbda * tr(eps[i]) * tr(eps[j])) * dV
		
		# Simplify expressions (switch comment on lines for reducing times)
		sintegr = sp.simplify(integr)
		# sintegr = integr

		# Store results in array
		bform1 = sp.symbols(f'element_matrix[{i*(4*3)+j}]')
		expr[i*4*3+j] = (ast.Assignment(bform1, sintegr))	

		# Take advantage of symmetry to reduce code-gen times
		if i != j:
			bform2 = sp.symbols(f'element_matrix[{i+(4*3)*j}]')
			expr[i+4*3*j] = (ast.Assignment(bform2, sintegr))

c_code(expr)
