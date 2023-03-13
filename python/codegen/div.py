#!/usr/bin/env python3

# ./div.py

from sfem_codegen import *

rf = ref_fun(qx, qy, qz)
dV = det3(A)

ux = coeffs('ux', 4)
uy = coeffs('uy', 4)
uz = coeffs('uz', 4)

f = fun(qx, qy, qz)


divu = 0.

# sum_i u_i * div(phi_i)
for i in range(0, 4):
	fi = f[i]
	dfdx = sp.diff(fi, qx)
	dfdy = sp.diff(fi, qy)
	dfdz = sp.diff(fi, qz)
	term = dfdx * ux[i] + dfdy * uy[i] + dfdz * uz[i]
	# c_log("------------")
	# c_log(term)
	divu = divu + term

divu = sp.simplify(divu)

# c_log(divu)

# Since divu is constant we can place it outside the integral
placeholer_divu = sp.symbols("pdivu")

expr = []
for i in range(0, 4):
	form = placeholer_divu * rf[i] * dV
	integr = sp.integrate(form, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1))

	bform = sp.symbols(f'element_vector[{i}]')
	expr.append(ast.Assignment(bform, sp.simplify(integr)))

for i in range(0, 4):
 	expr[i] = expr[i].subs(placeholer_divu, divu)

c_code(expr)

