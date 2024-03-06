#!/usr/bin/env python3

from sfem_symbolic import *

def generate_micro_kernel(dim):
	mu, lmbda, Gc, ls = sp.symbols('mu lambda Gc ls', real=True)

	# AT2
	def g(c):
		return (1 - c)**2

	def omega(c):
		return c**2

	def comega(c):
	 	return 2

	# -----------------------------------------
	# Symbolic 
	# -----------------------------------------

	u = coeffs('u', dim)
	u_grad = matrix_coeff('u_grad', dim, dim)
	c = sp.symbols('c')
	c_grad = coeffs('c_grad', dim)
	dV = sp.symbols('dV')

	def linear_strain(u_grad):
		return (u_grad + u_grad.T) / 2

	u_strain = linear_strain(u_grad)

	u_energy = lmbda/2 * tr(u_strain) * tr(u_strain) + mu * inner(u_strain, u_strain)
	c_energy = (Gc / comega(c)) * (omega(c)/ls + ls * inner(c_grad, c_grad))
	energy = (g(c) * u_energy + c_energy) * dV

	dedu = derivative1(energy, u_grad)
	dedc_grad = derivative1(energy, c_grad)
	dedc = derivative0(energy, c)

	trial_fun = trial_function("v")
	test_fun = test_function("v")

	trial_grad = trial_gradient("v", dim)
	test_grad  = test_gradient("v", dim)

	# creates dim basis function gradients with dim x dim size
	v_trial_grad = tensorize_vector(trial_grad)
	v_test_grad = tensorize_vector(test_grad)

	# -----------------------------------------
	# Linear form
	# -----------------------------------------

	u_form_1 = [0] * dim
	for d in range(0, dim):
		u_form_1[d] = inner(dedu, v_test_grad[d])

	c_form_1_grad = inner(dedc_grad, test_grad)
	c_form_1 = dedc * test_fun
	c_form1_all = c_form_1 + c_form_1_grad

	# -----------------------------------------

	form_2_uu = sp.Matrix(dim, dim, [0] * (dim * dim))
	form_2_cc = 0

	for d1 in range(0, dim):
		dfdu = derivative1(u_form_1[d1], u_grad)
		for d2 in range(0, dim):
			form_2_uu[d1, d2] = inner(dfdu, v_trial_grad[d2])

	form_2_uc = sp.Matrix(dim, 1, [0] * (dim))	
	for d1 in range(0, dim):
		dfdc = derivative0(u_form_1[d1], c)
		form_2_uc[d1, 0] = dfdc * trial_fun

		dfdc_grad = derivative1(u_form_1[d1], c_grad)
		form_2_uc[d1, 0] += inner(dfdc_grad, trial_grad)

	form_2_cu = sp.Matrix(1, dim, [0] * (dim))	
	for d1 in range(0, dim):
		dfdu = derivative1(c_form1_all, u_grad)
		form_2_cu[0, d1] = inner(dfdu, v_trial_grad[d1])

	form_2_cc = sp.Matrix(1, 1, [0] * (1))	
	form_2_cc[0, 0] = derivative0(c_form1_all, c) * trial_fun  
	form_2_cc[0, 0] += inner(derivative1(c_form1_all, c_grad), trial_grad)

	tpl = read_file('micro_kernel_isotropic_phase_field_tpl.h')
	code = tpl.format(
		MICRO_KERNEL_ENERGY=c_gen(form0_assign('element_scalar', energy)),
		MICRO_KERNEL_FORM1_U=c_gen(form1_assign('element_vector', u_form_1)),
		MICRO_KERNEL_FORM1_C=c_gen(form1_assign('element_vector', [c_form1_all])),
		MICRO_KERNEL_FORM2_UU=c_gen(form2_assign('element_matrix', form_2_uu)),
		MICRO_KERNEL_FORM2_UC=c_gen(form2_assign('element_matrix', form_2_uc)),
		MICRO_KERNEL_FORM2_CU=c_gen(form2_assign('element_matrix', form_2_cu)),
		MICRO_KERNEL_FORM2_CC=c_gen(form2_assign('element_matrix', form_2_cc)))

	str_to_file(f'micro_kernel_isotropic_phase_field_{dim}.h', code)

# def generate_kernel_code(dim):


if __name__ == '__main__':
	generate_micro_kernel(2)
	generate_micro_kernel(3)
