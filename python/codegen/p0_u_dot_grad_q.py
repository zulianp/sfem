#!/usr/bin/env python3

# ./div.py

from sfem_codegen import *

rf = ref_fun(qx, qy, qz)
dV = det3(A) / 6

ux = sp.symbols('ux')
uy = sp.symbols('uy')
uz = sp.symbols('uz')

u = [ux, uy, uz]
f = fun(qx, qy, qz)

expr = []

for i in range(0, 4):
	fi = f[i]
	dfdx = sp.diff(fi, qx)
	dfdy = sp.diff(fi, qy)
	dfdz = sp.diff(fi, qz)
	g = [dfdx, dfdy, dfdz]

	form = 0
	for d in range(0, 3):
		form += g[d] * u[d]

	integr = form * dV
	var = sp.symbols(f'element_vector[{i}]')
	expr.append(ast.Assignment(var, integr))

c_code(expr)

