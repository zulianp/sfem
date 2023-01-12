#!/usr/bin/env python3

from sfem_codegen import *

def n_test_functions():
	return 4*3

mu, lmbda = sp.symbols('mu lambda', real=True)

# Displacement

listdisp = []
for i in range(0, n_test_functions()):
	ui= sp.symbols(f'u[{i}]', real=True)
	listdisp.append(ui)

disp = sp.Matrix(n_test_functions(), 1, listdisp)

# Displacement gradient
du0dx, du0dy, du0dz = sp.symbols('du0dx du0dy d0udz', real=True)
du1dx, du1dy, du1dz = sp.symbols('du1dx du1dy d1udz', real=True)
du2dx, du2dy, du2dz = sp.symbols('du2dx du2dy d2udz', real=True)

gradu = sp.Matrix(3, 3, [
	du0dx, du0dy, du0dz, 
	du1dx, du1dy, du1dz, 
	du2dx, du2dy, du2dz]
)

def linear_strain(gradu):
	return (gradu + gradu.T) / 2

dV = det3(A) / 6
eps = symm_grad(qx, qy, qz)

shapegrad = tgrad(qx, qy, qz)

# Elastic energy
epsu = linear_strain(gradu)

e = lmbda/2 * tr(epsu) * tr(epsu) + mu * inner(epsu, epsu)

evalgradu = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0])

for i in range(0, n_test_functions()):
	for d1 in range(0, 3):
		for d2 in range(0, 3):
			evalgradu[d1, d2] += shapegrad[i][d1, d2] * disp[i]

def subsmat3x3(expr, oldmat, newmat):
	for d1 in range(0, 3):
		for d2 in range(0, 3):
			expr = expr.subs(oldmat[d1, d2], newmat[d1, d2])
	return expr

def makeenergy():
	integr = sp.simplify(e)
	integr = subsmat3x3(integr, gradu, evalgradu)

	# integr = sp.integrate(integr * det3(A), (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) # No need for this in linear tets
	integr = integr * dV

	integr = sp.simplify(integr)

	form = sp.symbols(f'element_energy')
	energy_expr = (ast.Assignment(form, integr))	
	return energy_expr

# Gradient
dedu = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0])

for d1 in range(0, 3):
	for d2 in range(0, 3):
		dedu[d1, d2] = sp.diff(e, gradu[d1, d2])

grade = [0]*n_test_functions()

for i in range(0, n_test_functions()):
	integr =  inner(dedu, eps[i])
	grade[i] = integr

def makegrad(i, q):
	integr =  inner(dedu, eps[i])
	integr = subsmat3x3(integr, gradu, evalgradu)

	# integr = sp.integrate(integr * det3(A), (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) # No need for this in linear tets
	integr = integr * dV
	
	integr = sp.simplify(integr)

	lform = sp.symbols(f'element_vector[{i}]')
	expr = ast.Assignment(lform, integr)
	q.put(expr)

def print_grads():
	for i in range(0, n_test_functions()):
		print(f'{i}) {grad_expr[i]}')

# Hessian
def makehessian(i, q):
	tuples = []

	He = sp.Matrix(3, 3, 
		[0, 0, 0, 
		 0, 0, 0,
		 0, 0, 0])

	for d1 in range(0, 3):
		for d2 in range(0, 3):
			He[d1, d2] = sp.diff(grade[i], gradu[d1, d2])

	for j in range(i, n_test_functions()):
		# Bilinear form
		integr = inner(He, eps[j]) 
		integr = subsmat3x3(integr, gradu, evalgradu)

		# integr = sp.integrate(integr * det3(A), (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) # No need for this in linear tets
		integr = integr * dV

		integr = sp.simplify(integr)

		# Store results in array
		bform1 = sp.symbols(f'element_matrix[{i * n_test_functions() + j}]')

		tuples.append((i, j, ast.Assignment(bform1, integr)))

		# Take advantage of symmetry to reduce code-gen times
		if i != j:
			bform2 = sp.symbols(f'element_matrix[{i + n_test_functions() * j}]')
			tuples.append((j, i, ast.Assignment(bform2, integr)))

	q.put(tuples)
