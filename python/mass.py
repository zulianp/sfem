#!/usr/bin/env python3

# import numpy as np
# import sympy as sp
# from sympy.utilities.codegen import codegen
# import sympy.codegen.ast as ast

from sfem_codegen import *

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
