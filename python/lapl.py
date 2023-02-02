#!/usr/bin/env python3

from sfem_codegen import *

simplify_expr = False

f = fun(qx, qy, qz)
dV = det3(A) / 6

listu = []
for i in range(0, 4):
	ui= sp.symbols(f'u[{i}]', real=True)
	listu.append(ui)

u = sp.Matrix(4, 1, listu)
grad_uh = sp.Matrix(3, 1, [0, 0, 0])

for i in range(0, 4):
	for d in range(0, 3):
		grad_uh[d] += sp.diff(f[i], q[d]) * u[i]

if False:
# if True:
	expr = []
	for i in range(0, 4):
		for j in range(0, 4):
			integr = 0
			for d in range(0, 3):
				integr += sp.diff(f[i], q[d]) * sp.diff(f[j], q[d]) * dV
			bform = sp.symbols(f'element_matrix[{i*4+j}]')

			if simplify_expr:
				integr = sp.simplify(integr)

			expr.append(ast.Assignment(bform, integr))
	print('---------------------------------------------------')
	c_code(expr)
	print('---------------------------------------------------')

if False:
# if True:
	expr = []
	for i in range(0, 4):
		integr = 0
		for d in range(0, 3):
			integr += grad_uh[d] * sp.diff(f[j], q[d]) * dV

		lform = sp.symbols(f'element_vector[{i}]')

		if simplify_expr:
			integr = sp.simplify(integr)

		expr.append(ast.Assignment(lform, integr))

	print('---------------------------------------------------')
	c_code(expr)
	print('---------------------------------------------------')

# if False:
if True:
	expr = []
	integr = 0
	for d in range(0, 3):
		integr += (grad_uh[d] **2)/2 

	integr *= dV

	form = sp.symbols(f'element_scalar[0]')

	if simplify_expr:
		integr = sp.simplify(integr)

	expr.append(ast.Assignment(form, integr))

	print('---------------------------------------------------')
	c_code(expr)
	print('---------------------------------------------------')
