#!/usr/bin/env python3

from sfem_codegen import *

simplify_expr = False

f = fun(qx, qy, qz)
rf = ref_fun(qx, qy, qz)
dV = det3(A) / 6

FFF = (Ainv * Ainv.T) * dV
cFFF = sp.Matrix(3, 3, [0]*9)

varidx = 0
for i in range(0, 3):
	for j in range(i, 3):
		var = sp.symbols(f'jac_inv[{varidx}*stride]')
		varidx += 1

		cFFF[i, j] = var
		cFFF[j, i] = var;


expr = []

for d1 in range(0, 3):
	for d2 in range(d1, 3):
		var = cFFF[d1, d2]
		val = FFF[d1, d2]
		# val = sp.simplify(val)
		expr.append(ast.Assignment(var, val))

c_code(expr)

def grad_phi(i):
	g = sp.Matrix(3, 1, [0, 0, 0])
	for d in range(0, 3):
		g[d] = sp.diff(rf[i], q[d])
	return g


expr = []

trial_gx = [0] * 4 
trial_gy = [0] * 4 
trial_gz = [0] * 4 


test_gx = [0] * 4 
test_gy = [0] * 4 
test_gz = [0] * 4 



for i in range(0, 4):
	g = grad_phi(i)
	test_gx[i] = g[0]
	test_gy[i] = g[1]
	test_gz[i] = g[2]
	FFFg = cFFF * g

	trial_gx[i] = FFFg[0]
	trial_gy[i] = FFFg[1]
	trial_gz[i] = FFFg[2]


expr = []
for i in range(0, 4):
	for j in range(0, 4):

		integr = 0
		
		integr += trial_gx[i] * test_gx[j]
		integr += trial_gy[i] * test_gy[j]
		integr += trial_gz[i] * test_gz[j]

		bform = sp.symbols(f'element_matrix[{i*4+j}*stride]')
		# bform_t = sp.symbols(f'element_matrix[{i+j*4}]')

		if simplify_expr:
			integr = sp.simplify(integr)

		expr.append(ast.Assignment(bform, integr))
		
		# if i != j:
		# 	expr.append(ast.Assignment(bform_t, bform))
print('---------------------------------------------------')
c_code(expr)
print('---------------------------------------------------')
