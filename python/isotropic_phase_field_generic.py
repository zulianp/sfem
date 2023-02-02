#!/usr/bin/env python3

from sfem_codegen import *

simplify_expr = False

def n_test_functions():
	return 4*4

mu, lmbda = sp.symbols('mu lambda', real=True)

# Displacement gradient
du0dx, du0dy, du0dz = sp.symbols('gradu[0] gradu[1] gradu[2]', real=True)
du1dx, du1dy, du1dz = sp.symbols('gradu[3] gradu[4] gradu[5]', real=True)
du2dx, du2dy, du2dz = sp.symbols('gradu[6] gradu[7] gradu[8]', real=True)

c, Gc, ls = sp.symbols('c Gc ls', real=True)
dcdx, dcdy, dcdz = sp.symbols('gradc[0] gradc[1] gradc[2]', real=True)
gradc = sp.Matrix(3, 1, [dcdx, dcdy, dcdz])

gradu = sp.Matrix(3, 3, [
	du0dx, du0dy, du0dz, 
	du1dx, du1dy, du1dz, 
	du2dx, du2dy, du2dz]
)

def linear_strain(gradu):
	return (gradu + gradu.T) / 2

dV = sp.symbols('dV')

trial_shapegrad = generic_grad('trial_grad')
test_shapegrad = generic_grad('test_grad')

trial_shapegrad3 = tensorize_grad(trial_shapegrad)
test_shapegrad3 = tensorize_grad(test_shapegrad)

trial_shape = sp.symbols('trial')
test_shape = sp.symbols('test')

# Elastic energy
epsu = linear_strain(gradu)
eu = lmbda/2 * tr(epsu) * tr(epsu) + mu * inner(epsu, epsu)

##############################
kernel_name = "isotropic_phase_field_AT2"
# AT2
def g(c):
	return (1 - c)**2

def omega(c):
	return c**2

def comega(c):
 	return 2
##############################

#############################################################
# Value
############################################################# 

ec = (Gc / comega(c)) * omega(c)/ls + ls * dot3(gradc, gradc)

e = g(c) * eu + ec

def makeenergy():
	integr = e * dV

	if simplify_expr:
		integr = sp.simplify(integr)

	form = sp.symbols(f'element_scalar[0]')
	energy_expr = (ast.AddAugmentedAssignment(form, integr))	
	return energy_expr

energy_expr = makeenergy()
energy_code = c_gen(energy_expr)

#############################################################
# Gradient
############################################################# 
# 
# # Displacement variable

dedu = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0])

for d1 in range(0, 3):
	for d2 in range(0, 3):
		dedu[d1, d2] = sp.diff(e, gradu[d1, d2])

grade_wrt_u = sp.Matrix(3, 1, 
	[0, 0, 0])


for d1 in range(0, 3):
	grade_wrt_u[d1] = inner(dedu, test_shapegrad3[d1])

# # Phase-field variable

# Derivative w.r.t to grad(c)
dedgradc = sp.Matrix(3, 1, 
	[0, 0, 0])

for d1 in range(0, 3):
	dedgradc[d1] = sp.diff(e, gradc[d1])

# Derivative w.r.t to c
dedc = sp.diff(e, c)

grade_wrt_c = dot3(dedgradc, test_shapegrad) + dedc * test_shape

# ####################################

def makegradu(i):
	integr = grade_wrt_u[i] * dV
	
	if simplify_expr:
		integr = sp.simplify(integr)

	lform = sp.symbols(f'element_vector[{i}]')
	expr = ast.AddAugmentedAssignment(lform, integr)
	return expr

def makegradc():
	integr = grade_wrt_c * dV
	
	if simplify_expr:
		integr = sp.simplify(integr)

	lform = sp.symbols(f'element_vector[{3}]')
	expr = ast.AddAugmentedAssignment(lform, integr)
	return expr

def makegrad():
	expr = [0] * 4
	for i in range(0, 3):
		expr[i] = makegradu(i)

	expr[3] = makegradc()
	return expr

grad_expr = makegrad()
gradient_code = c_gen(grad_expr)

#############################################################
# Hessian
############################################################# 

# uu
d2edu2 = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0])

Hessian_wrt_uu = sp.Matrix(3, 3, 
	[0, 0, 0, 
	 0, 0, 0,
	 0, 0, 0
	])

for d1 in range(0, 3):
	grade_wrt_u[d1] = inner(dedu, trial_shapegrad3[d1])

for i_trial in range(0, 3):
	for d1 in range(0, 3):
		for d2 in range(0, 3):
			d2edu2[d1, d2] = sp.diff(grade_wrt_u[i_trial], gradu[d1, d2])

	for i_test in range(0, 3):
		Hessian_wrt_uu[i_trial, i_test] = inner(d2edu2, test_shapegrad3[i_trial])

# uc / cu
d2edudc = sp.Matrix(3, 1, 
	[0, 0, 0])

Hessian_wrt_uc= sp.Matrix(3, 1, 
	[0, 0, 0
	])

for i_trial in range(0, 3):
	for d1 in range(0, 3):
		d2edudc[d1] = sp.diff(grade_wrt_u[i_trial], gradc[d1])

	Hessian_wrt_uc[i_trial] = dot3(d2edudc, test_shapegrad) + sp.diff(grade_wrt_u[i_test], c) * test_shape
# cc
grade_wrt_c = dot3(dedgradc, trial_shapegrad) + dedc * trial_shape
Hessian_wrt_cc = dot3(sp.diff(grade_wrt_c, gradc), test_shapegrad) + sp.diff(grade_wrt_c, c) * test_shape

def makehessianuu(i, j):
	integr = Hessian_wrt_uu[i, j] * dV

	if simplify_expr:
		integr = sp.simplify(integr)

	bform = sp.symbols(f'element_matrix[{i * 3 + j}]')
	return ast.AddAugmentedAssignment(bform, integr)

def makehessianuc(i):
	integr = Hessian_wrt_uc[i] * dV

	if simplify_expr:
		integr = sp.simplify(integr)

	bform = sp.symbols(f'element_matrix[{i * 4 + 3}]')
	return ast.AddAugmentedAssignment(bform, integr)

def makehessiancu(i):
	integr = Hessian_wrt_uc[i] * dV

	if simplify_expr:
		integr = sp.simplify(integr)

	bform = sp.symbols(f'element_matrix[{3 * 4 + i}]')
	return ast.AddAugmentedAssignment(bform, integr)

def makehessiancc():
	integr = Hessian_wrt_cc * dV

	if simplify_expr:
		integr = sp.simplify(integr)

	bform = sp.symbols(f'element_matrix[{3 * 4 + 3}]')
	return ast.AddAugmentedAssignment(bform, integr)

def makehessian():
	expr = [0] * (4 * 4)

	for i in range(0, 3):
		uc = makehessianuc(i)
		cu = makehessiancu(i)

		expr[i * 4 + 3] = uc
		expr[3 * 4 + i] = cu

		for j in range(0, 3):
			expr[i * 4 + j] = makehessianuu(i, j)

	expr[3*4 + 3] = makehessiancc()
	return expr

hessian_expr = makehessian()
hessian_code = c_gen(hessian_expr)

#############################################################
# Apply
############################################################# 

# Displacement gradient increment
ddelta_u0dx, ddelta_u0dy, ddelta_u0dz = sp.symbols('grad_delta_u[0] grad_delta_u[1] grad_delta_u[2]', real=True)
ddelta_u1dx, ddelta_u1dy, ddelta_u1dz = sp.symbols('grad_delta_u[3] grad_delta_u[4] grad_delta_u[5]', real=True)
ddelta_u2dx, ddelta_u2dy, ddelta_u2dz = sp.symbols('grad_delta_u[6] grad_delta_u[7] grad_delta_u[8]', real=True)

delta_c= sp.symbols('delta_c', real=True)
ddelta_cdx, ddelta_cdy, ddelta_cdz = sp.symbols('grad_delta_c[0] grad_delta_c[1] grad_delta_c[2]', real=True)
grad_delta_c = sp.Matrix(3, 1, [ddelta_cdx, ddelta_cdy, ddelta_cdz])

grad_delta_u = sp.Matrix(3, 3, 
	[ ddelta_u0dx, ddelta_u0dy, ddelta_u0dz,  
	  ddelta_u1dx, ddelta_u1dy, ddelta_u1dz, 
	  ddelta_u2dx, ddelta_u2dy, ddelta_u2dz ])


def makeapply():
	# Inputs are gradu, gradc, and c
	expr = None
	return expr

apply_expr = makeapply()
apply_code = c_gen(hessian_expr)


############################################################# 

params = """
	const real_t mu,
	const real_t lambda,
	const real_t Gc,
	const real_t ls,"""

args_eg = params + """
	const real_t c, 
	const real_t *gradc, 
	const real_t *gradu,
	const real_t test,
	const real_t trial,
	const real_t *test_grad,
	const real_t *trial_grad,
	const real_t dV,"""

args_H = params + """
	const real_t c,  
	const real_t *gradu,
	const real_t test,
	const real_t trial,
	const real_t *test_grad,
	const real_t *trial_grad,
	const real_t dV,
	real_t *element_matrix"""

tpl = """
// Basic includes
#include "sfem_base.h"
#include <math.h>

// Energy
void {kernel_name}_energy({args_e}) {{
{energy}
}}

// Gradient
void {kernel_name}_gradient({args_g}) {{
{gradient}
}}

// Hessian
void {kernel_name}_hessian({args_H}) {{
{hessian}
}}
"""

output = tpl.format(
	kernel_name = kernel_name,
	energy = energy_code, 
	gradient = gradient_code, 
	hessian = hessian_code, 
	args_e = args_eg + "\n\treal_t *element_scalar", 
	args_g = args_eg + "\n\treal_t *element_vector",  
	args_H = args_H)

console.print(output)

f = open('generated_code_temp.c', 'w')
f.write(output)
f.close()

