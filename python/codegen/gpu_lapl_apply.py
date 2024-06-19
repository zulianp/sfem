#!/usr/bin/env python3

from sfem_codegen import *

simplify_expr = False

f = fun(qx, qy, qz)
rf = ref_fun(qx, qy, qz)
dV = det3(A) / 6

FFF = (Ainv * Ainv.T) * dV
# FFF = (Ainv.T * Ainv) * dV
cFFF = sp.Matrix(3, 3, [0]*9)

varidx = 0
for i in range(0, 3):
	for j in range(i, 3):
		var = sp.symbols(f'fff[{varidx}*stride]')
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

left_gx = [0] * 4 
left_gy = [0] * 4 
left_gz = [0] * 4 


right_gx = [0] * 4 
right_gy = [0] * 4 
right_gz = [0] * 4 


for i in range(0, 4):
	g = grad_phi(i)
	right_gx[i] = g[0]
	right_gy[i] = g[1]
	right_gz[i] = g[2]

	FFFg = cFFF * g
	left_gx[i] = FFFg[0]
	left_gy[i] = FFFg[1]
	left_gz[i] = FFFg[2]


expr = []
for i in range(0, 4):
	for j in range(0, 4):

		integr = 0
		
		integr += left_gx[i] * right_gx[j]
		integr += left_gy[i] * right_gy[j]
		integr += left_gz[i] * right_gz[j]

		bform = sp.symbols(f'element_matrix[{i*4+j}*stride]')
		# bform_t = sp.symbols(f'element_matrix[{i+j*4}]')

		if simplify_expr:
			integr = sp.simplify(integr)

		expr.append(ast.Assignment(bform, integr))
		
print('---------------------------------------------------')
print('Hessian')
print('---------------------------------------------------')
c_code(expr)
print('---------------------------------------------------')

expr = []
for i in range(0, 4):
	integr = 0
	
	integr += left_gx[i] * right_gx[i]
	integr += left_gy[i] * right_gy[i]
	integr += left_gz[i] * right_gz[i]

	bform = sp.symbols(f'element_vector[{i}*stride]')

	if simplify_expr:
		integr = sp.simplify(integr)

	expr.append(ast.Assignment(bform, integr))
		
print('---------------------------------------------------')
print('diag(Hessian)')
print('---------------------------------------------------')
c_code(expr)
print('---------------------------------------------------')

u = coeffs('u', 4)

expr = []
for i in range(0, 4):
	integr = 0

	for j in range(0, 4):
		integr += left_gx[i] * right_gx[j] * u[j]
		integr += left_gy[i] * right_gy[j] * u[j]
		integr += left_gz[i] * right_gz[j] * u[j]

	lform = sp.symbols(f'element_vector[{i}*stride]')

	if simplify_expr:
		integr = sp.simplify(integr)

	expr.append(ast.Assignment(lform, integr))


print('---------------------------------------------------')
print('Gradient')
print('---------------------------------------------------')
c_code(expr)
print('---------------------------------------------------')