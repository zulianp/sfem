#!/usr/bin/env python3

from sfem_codegen import *

def n_test_functions():
	return 4*3

mu, lmbda = sp.symbols('mu lambda')

# Displacement
u0, u1, u2 = sp.symbols('u0 u1 u2')

u = sp.Matrix(3, 1, [u0, u1, u2])

# Displacement gradient
du0dx, du0dy, du0dz = sp.symbols('du0dx du0dy d0udz')
du1dx, du1dy, du1dz = sp.symbols('du1dx du1dy d1udz')
du2dx, du2dy, du2dz = sp.symbols('du2dx du2dy d2udz')

gradu = sp.Matrix(3, 3, [
	du0dx, du0dy, du0dz, 
	du1dx, du1dy, du1dz, 
	du2dx, du2dy, du2dz]
)

def linear_strain(gradu):
	return (gradu + gradu.T) / 2

dV = det3(A) / 6
eps = symm_grad(qx, qy, qz)

# Elastic energy
epsu = linear_strain(gradu)

e = lmbda/2 * tr(epsu) * tr(epsu) + mu * inner(epsu, epsu)

def makeenergy():
	integr = sp.simplify(e * det3(A))
	integr = sp.integrate(integr, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1))
	sintegr = sp.simplify(integr)

	form = sp.symbols(f'element_energy')
	energy_expr = (ast.Assignment(form, sintegr))	
	return energy_expr

# Gradient
dedu = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0])

for d1 in range(0, 3):
	for d2 in range(0, 3):
		dedu[d1, d2] = sp.diff(e, gradu[d1, d2])

grade = [0]*(4*3)

for i in range(0, 4*3):
	integr =  inner(dedu, eps[i])
	grade[i] = integr

def makegrad(i, q):
	integr =  inner(dedu, eps[i])
	grade[i] = integr

	# Simplify expressions (switch comment on lines for reducing times)
	integr = sp.integrate(integr * det3(A), (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1))
	sintegr = sp.simplify(integr)
	# sintegr = integr
	lform = sp.symbols(f'element_vector[{i}]')
	expr = ast.Assignment(lform, sintegr)
	q.put(expr)

def print_grads():
	for i in range(0, 4*3):
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

	for j in range(i, 4*3):
		# Bilinear form
		integr = inner(He, eps[j]) * dV
		
		# Simplify expressions (switch comment on lines for reducing times)
		sintegr = sp.simplify(integr)

		# Store results in array
		bform1 = sp.symbols(f'element_matrix[{i*(4*3)+j}]')

		tuples.append((i, j, ast.Assignment(bform1, sintegr)))

		# Take advantage of symmetry to reduce code-gen times
		if i != j:
			bform2 = sp.symbols(f'element_matrix[{i+(4*3)*j}]')
			tuples.append((j, i, ast.Assignment(bform2, sintegr)))

	q.put(tuples)
	