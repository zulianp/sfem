#!/usr/bin/env python3

from sfem_codegen import *

ux = coeffs('ux', 4)
uy = coeffs('uy', 4)
uz = coeffs('uz', 4)

u = [ux, uy, uz]
f = fun(qx, qy, qz)

divu = 0
for i in range(0, 4):
	fi = f[i]
	dfdx = sp.diff(fi, qx)
	dfdy = sp.diff(fi, qy)
	dfdz = sp.diff(fi, qz)
	g = [dfdx, dfdy, dfdz]

	for d in range(0, 3):
		divu += g[d] * u[d][i]

var = sp.symbols(f'element_value[0]')

expr = [ast.Assignment(var, divu)]
c_code(expr)
