#!/usr/bin/env python3

from sfem_codegen import *

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
