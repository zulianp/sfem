#!/usr/bin/env python3

from sfem_codegen import *

rf = ref_fun(qx, qy, qz)
dV = det3(A)

u = coeffs('u_p0', 1)

expr = []
for i in range(0, 4):
	form = u[0] * rf[i] * dV
	integr = sp.integrate(form, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1))

	lhs = sp.symbols(f'u_p1[{i}]')
	expr.append(ast.Assignment(lhs, sp.simplify(integr)))


for i in range(0, 4):
	weight = sp.integrate(rf[i] * dV, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1))

	lhs = sp.symbols(f'weight[{i}]')
	expr.append(ast.Assignment(lhs, sp.simplify(weight)))

c_code(expr)
