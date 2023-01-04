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

rf = ref_fun(qx, qy, qz)
dV = det3(A)

expr = []
sumterms=0
for i in range(0, 4):
	for j in range(0, 4):
		form = rf[i] * rf[j] * dV
		integr = sp.integrate(form, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1))
 
		bform = sp.symbols(f'element_matrix[{i*4+j}]')
		expr.append(ast.Assignment(bform, sp.simplify(integr)))
		sumterms+=integr

c_code(expr)

print("Test:")

test = sumterms.subs(x0, 0)
test = test.subs(x1, 1)
test = test.subs(x2, 0)
test = test.subs(x3, 0)

test = test.subs(y0, 0)
test = test.subs(y1, 0)
test = test.subs(y2, 1)
test = test.subs(y3, 0)

test = test.subs(z0, 0)
test = test.subs(z1, 0)
test = test.subs(z2, 0)
test = test.subs(z3, 1)

print(f'{test} = 1/6')
