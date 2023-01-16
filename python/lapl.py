#!/usr/bin/env python3

import numpy as np
import sympy as sp
from sympy.utilities.codegen import codegen
import sympy.codegen.ast as ast

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

def fun(x, y, z):
	xmb = x - b[0]
	ymb = y - b[1]
	zmb = z - b[2]

	xref = Ainv[0, 0] * xmb + Ainv[0, 1] * ymb  + Ainv[0, 2] * zmb
	yref = Ainv[1, 0] * xmb + Ainv[1, 1] * ymb  + Ainv[1, 2] * zmb
	zref = Ainv[2, 0] * xmb + Ainv[2, 1] * ymb  + Ainv[2, 2] * zmb
	return ref_fun(xref, yref, zref)

f = fun(qx, qy, qz)
dV = det3(A) / 6

expr = []

for i in range(0, 4):
	for j in range(0, 4):
		integr = 0
		for d in range(0, 3):
			integr += sp.diff(f[i], q[d]) * sp.diff(f[j], q[d]) * dV
		bform = sp.symbols(f'element_matrix[{i*4+j}]')
		expr.append(ast.Assignment(bform, sp.simplify(integr)))

c_code(expr)
