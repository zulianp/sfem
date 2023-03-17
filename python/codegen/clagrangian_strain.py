#!/usr/bin/env python3

from sfem_codegen import *

ux = coeffs('ux', 4)
uy = coeffs('uy', 4)
uz = coeffs('uz', 4)

f = fun(qx, qy, qz)

gradu = sp.Matrix(3, 3, [0.]*9)
Id = sp.Matrix(3, 3, [1, 0, 0,
					  0, 1, 0,
					  0, 0, 1])

for i in range(0, 4):
	fi = f[i]
	dfdx = sp.diff(fi, qx)
	dfdy = sp.diff(fi, qy)
	dfdz = sp.diff(fi, qz)
	
	gradu[0,0] += dfdx * ux[i]
	gradu[0,1] += dfdy * ux[i]
	gradu[0,2] += dfdz * ux[i]

	gradu[1,0] += dfdx * uy[i]
	gradu[1,1] += dfdy * uy[i]
	gradu[1,2] += dfdz * uy[i]

	gradu[2,0] += dfdx * uz[i]
	gradu[2,1] += dfdy * uz[i]
	gradu[2,2] += dfdz * uz[i]

F = Id + gradu
strain = (F.T * F - Id) / 2

# c_log(strain)

expr = []

strain0 = sp.symbols(f'strain[0]')
expr.append(ast.Assignment(strain0, strain[0,0]))

strain1 = sp.symbols(f'strain[1]')
expr.append(ast.Assignment(strain1, strain[0,1]))

strain2 = sp.symbols(f'strain[2]')
expr.append(ast.Assignment(strain2, strain[0, 2]))

strain3 = sp.symbols(f'strain[3]')
expr.append(ast.Assignment(strain3, strain[1, 1]))

strain4 = sp.symbols(f'strain[4]')
expr.append(ast.Assignment(strain4, strain[1, 2]))

strain5 = sp.symbols(f'strain[5]')
expr.append(ast.Assignment(strain5, strain[2, 2]))

c_code(expr)
