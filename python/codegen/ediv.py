#!/usr/bin/env python3

from sfem_codegen import *

rf = ref_fun(qx, qy, qz)
dV = det3(A)

ux = coeffs('ux', 4)
uy = coeffs('uy', 4)
uz = coeffs('uz', 4)

u = [ux, uy, uz]
f = fun(qx, qy, qz)

var_f = sp.symbols("f")
integr = sp.integrate(var_f, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) * dV 
sum_f = 0

expr = []
for i in range(0, 4):
	fi = f[i]
	dfdx = sp.diff(fi, qx)
	dfdy = sp.diff(fi, qy)
	dfdz = sp.diff(fi, qz)
	g = [dfdx, dfdy, dfdz]

	for d in range(0, 3):
		sum_f += integr.subs(var_f, -g[d] * u[d][i])

var = sp.symbols(f'element_value[0]')
lform = integr.subs(var_f, sum_f)
expr.append(ast.Assignment(var, sp.simplify(lform)))

c_code(expr)
