#!/usr/bin/env python3

from sfem_codegen import *

# simplify_expr = False
simplify_expr = True

def n_test_functions():
	return 4*4

mu, lmbda = sp.symbols('mu lambda', real=True)

# Displacement
listdisp = []

for i in range(0, n_test_functions()):
	ui= sp.symbols(f'u[{i}]', real=True)
	listdisp.append(ui)

disp = sp.Matrix(n_test_functions(), 1, listdisp)

# Phase
listphase = []

for i in range(0, 4):
	ci= sp.symbols(f'c[{i}]', real=True)
	listphase.append(ci)

phase = sp.Matrix(len(listphase), 1, listphase)

# Displacement gradient
du0dx, du0dy, du0dz = sp.symbols('du0dx du0dy d0udz', real=True)
du1dx, du1dy, du1dz = sp.symbols('du1dx du1dy d1udz', real=True)
du2dx, du2dy, du2dz = sp.symbols('du2dx du2dy d2udz', real=True)

c, Gc, ls = sp.symbols('varc Gc ls', real=True)
dcdx, dcdy, dcdz = sp.symbols('dcdx dcdy dcdz')
gradc = sp.Matrix(3, 1, [dcdx, dcdy, dcdz])

gradu = sp.Matrix(3, 3, [
	du0dx, du0dy, du0dz, 
	du1dx, du1dy, du1dz, 
	du2dx, du2dy, du2dz]
)

def linear_strain(gradu):
	return (gradu + gradu.T) / 2

dV = qw * det3(A) / 6
eps = symm_grad(qx, qy, qz)

shapegrad = tgrad(qx, qy, qz)

# Elastic energy
epsu = linear_strain(gradu)
eu = lmbda/2 * tr(epsu) * tr(epsu) + mu * inner(epsu, epsu)

##############################
# AT2
def g(c):
	return (1 - c)**2

def omega(c):
	return c**2

def comega(c):
 	return 2
##############################

ec = (Gc / comega(c)) * omega(c)/ls + ls * dot3(gradc, gradc)

e = g(c) * eu + ec

#############################################

evalgradu = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0])

for i in range(0, 4*3):
	for d1 in range(0, 3):
		for d2 in range(0, 3):
			evalgradu[d1, d2] += shapegrad[i][d1, d2] * disp[i]

#############################################

evalc = 0
rf = ref_fun(qx, qy, qz)

for i in range(0, 4):
	evalc += phase[i]  * rf[i]

evalgradc = sp.Matrix(3, 1, [0, 0, 0])

# Global coordinates (!!! qx, qy, qz are eliminated by diff because f is linear, so this kinda ok)
f = fun(qx, qy, qz)
for i in range(0, 4):
	for d1 in range(0, 3):
		evalgradc[d2] += phase[i] * sp.diff(f[i], q[d1])

# evalgradc = sp.simplify(evalgradc)

#############################################

def makeenergy():
	print(e)
	print(sp.collect(e, c))

	integr = e
	integr = subsmat3x3(integr, gradu, evalgradu)

	integr = integr.subs(c, evalc)
	integr = subsvec3(integr, gradc, evalgradc)
	
	# integr = sp.integrate(integr * det3(A), (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1))
	integr = integr * dV

	if simplify_expr:
		integr = sp.simplify(integr)

	form = sp.symbols(f'element_energy')
	energy_expr = (ast.Assignment(form, integr))	
	return energy_expr

# energy_expr = makeenergy()
# c_code(energy_expr)

####################################
# Gradient
####################################

# Displacement variable

dedu = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0])

for d1 in range(0, 3):
	for d2 in range(0, 3):
		dedu[d1, d2] = sp.diff(e, gradu[d1, d2])

grade_wrt_u = [0]*(4*3)

for i in range(0, (4*3)):
	integr =  inner(dedu, shapegrad[i])
	grade_wrt_u[i] = integr

# Phase-field variable

# Derivative w.r.t to grad(c)
dedgradc = sp.Matrix(3, 1, 
	[0, 0, 0])

for d1 in range(0, 3):
	dedgradc[d1] = sp.diff(e, gradc[d1])

# Derivative w.r.t to c
dedc = sp.diff(e, c)

grade_wrt_c = [0]*(4)

for i in range(0, 4):
	integr = 0.

	for d1 in range(0, 3):
		integr += dedgradc[d1] * sp.diff(f[i], q[d1])
	
	grade_wrt_c[i] = integr + (dedc * rf[i])

####################################

def makegradu(i, q):
	integr =  grade_wrt_u[i]
	integr = subsmat3x3(integr, gradu, evalgradu)

	integr = integr.subs(c, evalc)
	integr = subsvec3(integr, gradc, evalgradc)

	# integr = sp.integrate(integr * det3(A), (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) # No need for this in linear tets
	integr = integr * dV
	
	if simplify_expr:
		integr = sp.simplify(integr)

	lform = sp.symbols(f'element_vector[{i}]')
	expr = ast.Assignment(lform, integr)

	q.put(expr)
	
def makegradc(i, q):
	assert(i < 4)
	integr =  grade_wrt_c[i]

	integr = subsmat3x3(integr, gradu, evalgradu)

	integr = integr.subs(c, evalc)
	integr = subsvec3(integr, gradc, evalgradc)

	# integr = sp.integrate(integr * det3(A), (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) # No need for this in linear tets
	integr = integr * dV
	
	if simplify_expr:
		integr = sp.simplify(integr)

	lform = sp.symbols(f'element_vector[{(4*3) + i}]')
	expr = ast.Assignment(lform, integr)

	q.put(expr)

def makegrad(i, q):
	if i < 4*3:
		makegradu(i, q)
	else:
		makegradc(i - (4*3), q)

	print('Done!')

# def print_grads():
# 	for i in range(0, (4*3)):
# 		print(f'{i}) {grad_expr[i]}')

# Hessian
def makehessian(i, q):
	tuples = []

	# He = sp.Matrix(3, 3, 
	# 	[0, 0, 0, 
	# 	 0, 0, 0,
	# 	 0, 0, 0])

	# for d1 in range(0, 3):
	# 	for d2 in range(0, 3):
	# 		He[d1, d2] = sp.diff(grade[i], gradu[d1, d2])

	# for j in range(i, (4*3)):
	# 	# Bilinear form
	# 	integr = inner(He, shapegrad[j]) 
	# 	integr = subsmat3x3(integr, gradu, evalgradu)

	# 	# integr = sp.integrate(integr * det3(A), (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) # No need for this in linear tets
	# 	integr = integr * dV

	# 	if simplify_expr:
	# 		integr = sp.simplify(integr)

	# 	# Store results in array
	# 	bform1 = sp.symbols(f'element_matrix[{i * (4*3) + j}]')

	# 	tuples.append((i, j, ast.Assignment(bform1, integr)))

	# 	# Take advantage of symmetry to reduce code-gen times
	# 	if i != j:
	# 		bform2 = sp.symbols(f'element_matrix[{i + (4*3) * j}]')
	# 		tuples.append((j, i, ast.Assignment(bform2, integr)))

	q.put(tuples)
