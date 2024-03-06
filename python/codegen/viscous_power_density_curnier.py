#!/usr/bin/env python3

from sfem_codegen import *
from vonmises import *

simplify_expr = False

def n_test_functions():
	return 4*3

mu, lmbda, eta, dt = sp.symbols('mu lambda eta dt', real=True)

# Displacement


disp = coeffs('u', n_test_functions())
disp_old = coeffs('u_old', n_test_functions())


# Displacement gradient
F00, F01, F02 = sp.symbols('F00 F01 F02', real=True)
F10, F11, F12 = sp.symbols('F10 F11 F12', real=True)
F20, F21, F22 = sp.symbols('F20 F21 F22', real=True)

F = sp.Matrix(3, 3, [
	F00, F01, F02, 
	F10, F11, F12, 
	F20, F21, F22]
)

F_old_00, F_old_01, F_old_02 = sp.symbols('F_old_00 F_old_01 F_old_02', real=True)
F_old_10, F_old_11, F_old_12 = sp.symbols('F_old_10 F_old_11 F_old_12', real=True)
F_old_20, F_old_21, F_old_22 = sp.symbols('F_old_20 F_old_21 F_old_22', real=True)

F_old = sp.Matrix(3, 3, [
	F_old_00, F_old_01, F_old_02, 
	F_old_10, F_old_11, F_old_12, 
	F_old_20, F_old_21, F_old_22]
)

Fp_00, Fp_01, Fp_02 = sp.symbols('Fp_00 Fp_01 Fp_02', real=True)
Fp_10, Fp_11, Fp_12 = sp.symbols('Fp_10 Fp_11 Fp_12', real=True)
Fp_20, Fp_21, Fp_22 = sp.symbols('Fp_20 Fp_21 Fp_22', real=True)

Fp = sp.Matrix(3, 3, [
	Fp_00, Fp_01, Fp_02, 
	Fp_10, Fp_11, Fp_12, 
	Fp_20, Fp_21, Fp_22]
)


Cp = Fp.T * F + F.T * Fp
W = eta / 2 * tr(Cp * Cp)

time_integr_Fp = (F - F_old)/dt


# dWdFp = sp.zeros(3, 3)
# for d1 in range(0, 3):
# 	for d2 in range(0, 3):
# 		dWdFp[d1, d2] = sp.diff(W, Fp[d1, d2])

# 1) Derive Cp with Fp (to compute gradient of viscous energy)
# 2) Subs Fp with time derivative e.g., (F - F_old)/dt
# 3) Compute linearization with respect F for the hessian

d = 3
J = det3(F)
logJ = sp.log(J) 
F_inv = inv3(F)
F_inv_t = F_inv.T
C = F.T * F
I_C = tr(C)

shapegrad = tgrad(qx, qy, qz)
dV = det3(A) / 6

# # Elastic energy
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

evalF_old = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0])

# Displacement gradient
for i in range(0, n_test_functions()):
	for d1 in range(0, 3):
		for d2 in range(0, 3):
			evalF_old[d1, d2] += shapegrad[i][d1, d2] * disp_old[i]

# # Add identity
for d1 in range(0, 3):
	evalF[d1, d1] += 1

def subsmat3x3(expr, oldmat, newmat):
	for d1 in range(0, 3):
		for d2 in range(0, 3):
			expr = expr.subs(oldmat[d1, d2], newmat[d1, d2])
	return expr

def makeenergy():
	integr = sp.simplify(e)
	integr += sp.simplify(W)
	integr = subsmat3x3(integr, Fp, time_integr_Fp)
	integr = subsmat3x3(integr, F, evalF)
	integr = subsmat3x3(integr, F_old, evalF_old)
	integr = integr * dV

	if simplify_expr:
		integr = sp.simplify(integr)

	form = sp.symbols(f'element_energy')
	energy_expr = (ast.Assignment(form, integr))	
	return energy_expr

# c_code(makeenergy())

# Gradient
dedF = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0])

for d1 in range(0, 3):
	for d2 in range(0, 3):
		dedF[d1, d2] = sp.diff(e, F[d1, d2]) + sp.diff(W, Fp[d1, d2])
		dedF[d1, d2] = subsmat3x3(dedF[d1, d2], Fp, time_integr_Fp)

grade = [0]*n_test_functions()

for i in range(0, n_test_functions()):
	integr =  inner(dedF, shapegrad[i])
	grade[i] = integr

def makegrad(i, q):
	integr =  grade[i]
	integr = subsmat3x3(integr, F, evalF)
	integr = subsmat3x3(integr, F_old, evalF_old)
	integr = integr * dV
	
	if simplify_expr:
		integr = sp.simplify(integr)

	lform = sp.symbols(f'element_vector[{i}]')
	expr = ast.Assignment(lform, integr)
	q.put(expr)

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
		integr = subsmat3x3(integr, F_old, evalF_old)

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
