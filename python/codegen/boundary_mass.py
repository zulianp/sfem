#!/usr/bin/env python3

from sfem_codegen import *

S = sp.Matrix(3, 2, [
	 x1 - x0, x2 - x0,
	 y1 - y0, y2 - y0,
	 z1 - z0, z2 - z0
	])

StS = S.T * S
dS = sp.sqrt(det2(StS))
rf = [1 - qx - qy, qx, qy]

expr = []

for i in range(0, 3):
	for j in range(0, 3):
		f = rf[i] * rf[j]
		integr = sp.integrate(f, (qy, 0, 1 - qx), (qx, 0, 1)) * dS
		res = sp.symbols(f'element_matrix[{i * 3 + j}]')
		expr.append(ast.Assignment(res, sp.simplify(integr)))

c_code(expr)
