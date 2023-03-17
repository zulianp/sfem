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

placeholer_grad_phi = sp.symbols("grad_phi")

expr = []
for i in range(0, 4):
	fi = f[i]
	dfdx = sp.diff(fi, qx)
	dfdy = sp.diff(fi, qy)
	dfdz = sp.diff(fi, qz)
	g = [dfdx, dfdy, dfdz]

	lform = 0
	for j in range(0, 4):
		form = placeholer_grad_phi * rf[j] 
		integr = sp.integrate(form, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) 
		
		for d in range(0, 3):
			lform += integr.subs(placeholer_grad_phi, -g[d] * u[d][j])

	lform *= dV
	
	var = sp.symbols(f'element_vector[{i}]')
	expr.append(ast.Assignment(var, sp.simplify(lform)))

c_code(expr)

