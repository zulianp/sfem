#!/usr/bin/env python3

from sfem_codegen import *

simplify_expr = False

f = fun(qx, qy, qz)
rf = ref_fun(qx, qy, qz)
dV = det3(A) / 6

cainv = sp.Matrix(3, 3, [0]*9)
for i in range(0, 3):
	for j in range(0, 3):
		cainv[i, j] = sp.symbols(f'jac_inv[{i*3+j}*stride]')


cdv = sp.symbols('dv')


expr = []
for d1 in range(0, 3):
	for d2 in range(0, 3):
		var = cainv[d1, d2]
		expr.append(ast.Assignment(var, Ainv[d1, d2]))

c_code(expr)

expr = []
expr.append(ast.Assignment(cdv, (1./det3(cainv))/6))

c_code(expr)

def grad_phi(i):
	g = sp.Matrix(3, 1, [0, 0, 0])
	for d in range(0, 3):
		g[d] = sp.diff(rf[i], q[d])

	return g.T * cainv

c_code(expr)

expr = []

cgx = coeffs("gx", 4) 
cgy = coeffs("gy", 4) 
cgz = coeffs("gz", 4) 

for i in range(0, 4):
	g = grad_phi(i)
	cgx[i] = g[0]
	cgy[i] = g[1]
	cgz[i] = g[2]


expr = []
for i in range(0, 4):
	for j in range(0, 4):

		integr = 0
		
		integr += cgx[i] * cgx[j]
		integr += cgy[i] * cgy[j]
		integr += cgz[i] * cgz[j]
		
		integr *= cdv

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
