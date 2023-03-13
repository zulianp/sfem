#!/usr/bin/env python3

from sfem_codegen import *

simplify_expr = False

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
F00, F01, F02 = sp.symbols('F00 F01 F02', real=True)
F10, F11, F12 = sp.symbols('F10 F11 F12', real=True)
F20, F21, F22 = sp.symbols('F20 F21 F22', real=True)

F = sp.Matrix(3, 3, [
	F00, F01, F02, 
	F10, F11, F12, 
	F20, F21, F22]
)

d = 3
J = det3(F)
logJ = sp.log(J) 
F_inv = inv3(F)
F_inv_t = F_inv.T
C = F.T * F
I_C = tr(C)

shapegrad = tgrad(qx, qy, qz)
dV = det3(A) / 6

# Elastic energy
e = mu/2 *(I_C - d) - mu * logJ + (lmbda/2) * logJ**2

evalF = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0])

# Displacement gradient
for i in range(0, n_test_functions()):
	for d1 in range(0, 3):
		for d2 in range(0, 3):
			evalF[d1, d2] += shapegrad[i][d1, d2] * disp[i]

# Add identity
for d1 in range(0, 3):
	evalF[d1, d1] += 1

def subsmat3x3(expr, oldmat, newmat):
	for d1 in range(0, 3):
		for d2 in range(0, 3):
			expr = expr.subs(oldmat[d1, d2], newmat[d1, d2])
	return expr

def makeenergy():
	integr = sp.simplify(e)
	integr = subsmat3x3(integr, F, evalF)

	# integr = sp.integrate(integr * det3(A), (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) # No need for this in linear tets
	integr = integr * dV

	if simplify_expr:
		integr = sp.simplify(integr)

	form = sp.symbols(f'element_energy')
	energy_expr = (ast.Assignment(form, integr))	
	return energy_expr

# Gradient
dedF = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0])

for d1 in range(0, 3):
	for d2 in range(0, 3):
		dedF[d1, d2] = sp.diff(e, F[d1, d2])

grade = [0]*n_test_functions()

for i in range(0, n_test_functions()):
	integr =  inner(dedF, shapegrad[i])
	grade[i] = integr

def makegrad(i, q):
	integr =  grade[i]
	integr = subsmat3x3(integr, F, evalF)

	# integr = sp.integrate(integr * det3(A), (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) # No need for this in linear tets
	integr = integr * dV
	
	if simplify_expr:
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
			He[d1, d2] = sp.diff(grade[i], F[d1, d2])

	for j in range(i, n_test_functions()):
		# Bilinear form
		integr = inner(He, shapegrad[j]) 
		integr = subsmat3x3(integr, F, evalF)

		# integr = sp.integrate(integr * det3(A), (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) # No need for this in linear tets
		integr = integr * dV

		if simplify_expr:
			integr = sp.simplify(integr)

		# Store results in array
		bform1 = sp.symbols(f'element_matrix[{i * n_test_functions() + j}]')

		tuples.append((i, j, ast.Assignment(bform1, integr)))

		# Take advantage of symmetry to reduce code-gen times
		if i != j:
			bform2 = sp.symbols(f'element_matrix[{i + n_test_functions() * j}]')
			tuples.append((j, i, ast.Assignment(bform2, integr)))

	q.put(tuples)
