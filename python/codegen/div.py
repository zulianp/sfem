#!/usr/bin/env python3

# ./div.py

from sfem_codegen import *

rf = ref_fun(qx, qy, qz)
dV = det3(A)

ux = coeffs('ux', 4)
uy = coeffs('uy', 4)
uz = coeffs('uz', 4)

u = [ux, uy, uz]
f = fun(qx, qy, qz)

divu = 0.
for i in range(0, 4):
	fi = f[i]
	dfdx = sp.diff(fi, qx)
	dfdy = sp.diff(fi, qy)
	dfdz = sp.diff(fi, qz)
	g = [dfdx, dfdy, dfdz]

	for d in range(0, 3):
		divu += g[d] * u[d][i]


var_f = sp.symbols("var_f")

expr = []
for i in range(0, 4):
	lform = rf[i] * var_f
	integr = sp.integrate(lform, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) * dV 
	integr = integr.subs(var_f, divu)

	var = sp.symbols(f'element_vector[{i}]')
	expr.append(ast.Assignment(var, integr))

c_code(expr)

