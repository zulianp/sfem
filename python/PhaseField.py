#!/usr/bin/env python3

from sfem_codegen import *

simplify_expr = False
# simplify_expr = True

# class Tet4:
# 	def __init__(self):
# 		self.type = 'TET4'
# 		self.dim = 3

class GenericFE:
	def __init__(self, name, dim):
		self.type = 'GENERIC'
		self.dim = dim

		self.value_ = sp.symbols(name)

		temp_grad = []
		for d in range(0, dim):
			s = sp.symbols(f'grad_{name}[{d}]')
			temp_grad.append(s)

		self.grad_ = sp.Matrix(dim, 1, temp_grad)

	def is_generic(self):
		return True 

	def grad(self):
		return self.grad_

	def value(self):
		return self.value_

class TensorProductFE:
	def __init__(self, elem_type, block_size):
		self.elem_type = elem_type
		self.block_size = block_size


		temp_grad = []

		if(elem_type.is_generic()):
			g = elem_type.grad()

			for d1 in range(0, block_size):
				z = [0]*(block_size * elem_type.dim)
				G = sp.Matrix(block_size, elem_type.dim, z)

				for d2 in range(0, elem_type.dim):
					G[d1, d2] = g[d2]

				temp_grad.append(G)

			self.grad_ = temp_grad

		else:
			# Implement me
			assert False

	def is_generic(self):
		return self.elem_type.is_generic() 

	def grad(self):
		return self.grad_

class Function:
	def __init__(self, name, elem_type):
		self.name = name
		self.grad_name = f'grad_{name}'
		self.elem_type = elem_type
		self.block_size = 1

		self.value_ = sp.symbols(name, real=True)

		grad_list = []

		dim = elem_type.dim
		for d1 in range(0, dim):
			s = sp.symbols(f'{self.grad_name}[{d1}]', real=True)
			grad_list.append(s)

		self.grad_ = sp.Matrix(dim, 1, grad_list)

	def grad(self):
		return self.grad_

	def value(self):
		return self.value_

	def elem(self):
		return self.elem_type

class VectorFunction:
	def __init__(self, name, elem_type, block_size):
		self.name = name
		self.grad_name = f'grad_{name}'
		self.block_size = block_size
		self.elem_type = TensorProductFE(elem_type, block_size)

		dim = elem_type.dim


		grad_list = []
		for d1 in range(0, block_size):
			for d2 in range(0, elem_type.dim):
				s = sp.symbols(f'{self.grad_name}[{d1*elem_type.dim + d2}]', real=True)
				grad_list.append(s)

		self.grad_ = sp.Matrix(block_size, dim, grad_list)

	def grad(self):
		return self.grad_


def is_matrix(expr):
	return sp.matrices.immutable.ImmutableDenseMatrix == type(expr) or sp.matrices.dense.MutableDenseMatrix == type(expr)

def derivative(f, var):
	if is_matrix(var):
		rows, cols = var.shape
		z = [0]*(rows*cols)
		ret = sp.Matrix(rows, cols, z)

		for d1 in range(0, rows):
			for d2 in range(0, cols):
				ret[d1, d2] = sp.diff(f, var[d1, d2])
		return ret
	else:
		return sp.diff(f, var)

def directional_derivative(f, var, h):
	deriv = derivative(f, var)
	return inner(deriv, h)

class Model:
	def __init__(self, elem_trial, elem_test):
		self.elem_trial = elem_trial
		self.elem_test = elem_test
		self.dim = elem_test.dim

		self.dV = sp.symbols('dV')

	def set_energy(self, e):
		self.energy_ = e

	def energy(self):
		return self.energy_

	def makeenergy(self):
		integr = self.energy() * self.dV

		if simplify_expr:
			integr = sp.simplify(integr)

		form = sp.symbols(f'element_scalar[0]')
		energy_expr = (ast.AddAugmentedAssignment(form, integr))	
		return energy_expr

class PhaseFieldBase(Model):
	def __init__(self, elem_trial, elem_test):
		super().__init__(elem_trial, elem_test)
		dim  = elem_test.dim
		self.elem_test = elem_test
		self.tp_element_test = TensorProductFE(elem_test, dim)

		self.phase = Function('c', elem_trial)
		self.displacement = VectorFunction('u', elem_trial, dim)
	
	def initialize(self, energy):
		self.set_energy(energy)

		trial = self.phase.elem_type
		test = self.elem_test
		phase = self.phase

		displacement = self.displacement
		tp_test = self.tp_element_test 

		dtrial = self.displacement.elem_type.grad()
		dtest = tp_test.grad()
		n = len(dtest)
		assert(n == tp_test.block_size)


		dedc = derivative(energy, phase.value())
		dedgradc = derivative(energy, phase.grad())

		# Gradient
		self.grad_wrt_c = dedc * test.value() + inner(dedgradc, test.grad())

		dedu = derivative(energy, displacement.grad())
		self.grad_wrt_u = sp.Matrix(n, 1, [0]*n)

		for d1 in range(0, n):
			self.grad_wrt_u[d1] = inner(dedu, dtest[d1])	

		# Hessian
		self.hessian_wrt_uu = sp.Matrix(n, n, [0]*(n*n))

		H_grad_u = sp.Matrix(n, 1, [0]*n)
		for d1 in range(0, n):
			H_grad_u[d1] = inner(dedu, dtrial[d1])

		for i_trial in range(0, n):
			d2edu2 = derivative(H_grad_u[i_trial], displacement.grad())

			for i_test in range(0, n):
				self.hessian_wrt_uu[i_trial, i_test] = inner(d2edu2, dtest[i_test])

		self.hessian_wrt_uc = sp.Matrix(n, 1, [0] * n)

		for i_trial in range(0, n):
			d2edudc = derivative(H_grad_u[i_trial], phase.grad())
			self.hessian_wrt_uc[i_trial] = inner(d2edudc, test.grad()) + sp.diff(H_grad_u[i_trial], phase.value()) * test.value()
		
		# cc
		grad_wrt_c = inner(dedgradc, trial.grad()) + dedc * trial.value()
		self.hessian_wrt_cc = inner(derivative(grad_wrt_c, phase.grad()), test.grad()) + derivative(grad_wrt_c, phase.value()) * test.value()

	def makegradu(self, i):
		dV = self.dV
		integr = self.grad_wrt_u[i] * dV
		
		if simplify_expr:
			integr = sp.simplify(integr)

		lform = sp.symbols(f'element_vector[{i}]')
		expr = ast.AddAugmentedAssignment(lform, integr)
		return expr

	def makegradc(self, i):
		dV = self.dV
		integr = self.grad_wrt_c * dV
		
		if simplify_expr:
			integr = sp.simplify(integr)

		lform = sp.symbols(f'element_vector[{i}]')
		expr = ast.AddAugmentedAssignment(lform, integr)
		return expr

	def makegrad(self):
		ndisp = self.tp_element_test.block_size
		n = ndisp + 1

		expr = [0] * n

		for i in range(0, ndisp):
			expr[i] = self.makegradu(i)

		expr[ndisp] = self.makegradc(ndisp)
		return expr

	def makehessianuu(self, i, j):
		n = self.tp_element_test.block_size + 1
		dV = self.dV

		integr = self.hessian_wrt_uu[i, j] * dV

		if simplify_expr:
			integr = sp.simplify(integr)

		bform = sp.symbols(f'element_matrix[{i * n + j}]')
		return ast.AddAugmentedAssignment(bform, integr)

	def makehessianuc(self, i, j):
		n = self.tp_element_test.block_size + 1
		dV = self.dV

		integr = self.hessian_wrt_uc[i] * dV

		if simplify_expr:
			integr = sp.simplify(integr)

		bform = sp.symbols(f'element_matrix[{i * n + j}]')
		return ast.AddAugmentedAssignment(bform, integr)

	def makehessiancu(self, j, i):
		n = self.tp_element_test.block_size + 1
		dV = self.dV

		integr = self.hessian_wrt_uc[i] * dV

		if simplify_expr:
			integr = sp.simplify(integr)

		bform = sp.symbols(f'element_matrix[{j * n + i}]')
		return ast.AddAugmentedAssignment(bform, integr)

	def makehessiancc(self, i):
		n = self.tp_element_test.block_size + 1
		dV = self.dV

		integr = self.hessian_wrt_cc * dV

		if simplify_expr:
			integr = sp.simplify(integr)

		bform = sp.symbols(f'element_matrix[{i * n + i}]')
		return ast.AddAugmentedAssignment(bform, integr)

	def makehessian(self):
		ndisp = self.tp_element_test.block_size
		n = ndisp + 1

		expr = [0] * (n * n)

		for i in range(0, ndisp):
			uc = self.makehessianuc(i, ndisp)
			cu = self.makehessiancu(ndisp, i)

			expr[i * n + ndisp] = uc
			expr[ndisp * n + i] = cu

			for j in range(0, ndisp):
				expr[i * n + j] = self.makehessianuu(i, j)

		expr[ndisp * n + ndisp] = self.makehessiancc(ndisp)
		return expr

	def make_split_hessian_uu(self):
		ndisp = self.tp_element_test.block_size

		expr = [0] * (ndisp * ndisp)
		for i in range(0, ndisp):
			for j in range(0, ndisp):
				expr[i * ndisp + j] = self.makehessianuu(i, j)

		return expr

	def genreate_split_code(self):
		# Split operators
		gradient_c_expr = self.makegradc(0)	
		gradient_c_code = c_gen(gradient_c_expr)

		ndisp = self.tp_element_test.block_size
		gradient_u_expr = []
		
		for d1 in range(0, ndisp):
			gradient_u_expr.append(self.makegradu(d1))

		gradient_u_code = c_gen(gradient_u_expr)

		console.print(gradient_c_code)
		console.print(gradient_u_code)


		hessian_uu_expr = self.make_split_hessian_uu()
		hessian_uu_code = c_gen(hessian_uu_expr)

		console.print(hessian_uu_code)


	def generate_monolithic_code(self):
		energy_expr = self.makeenergy()
		energy_code = c_gen(energy_expr)

		gradient_expr = self.makegrad()
		gradient_code = c_gen(gradient_expr)

		hessian_expr = self.makehessian()
		hessian_code = c_gen(hessian_expr)

		params = ""
		for p in self.params:
			params += f"const real_t {p},\n"

		args_e = params + f"""const real_t {self.phase.name}, 
			const real_t *{self.phase.grad_name}, 
			const real_t *{self.displacement.grad_name},
			const real_t dV,
			real_t *element_scalar
			"""

		args_g = params + f"""const real_t {self.phase.name}, 
			const real_t *{self.phase.grad_name}, 
			const real_t *{self.displacement.grad_name},
			const real_t test,
			const real_t trial,
			const real_t *test_grad,
			const real_t dV,
			real_t *element_vector
			"""

		args_H = params + f"""const real_t {self.phase.name},  
			const real_t *{self.displacement.grad_name},
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
			kernel_name = self.kernel_name,
			energy = energy_code, 
			gradient = gradient_code, 
			hessian = hessian_code, 
			args_e = args_e,
			args_g = args_g,
			args_H = args_H)

		return output


	def generate_code(self):
		output = self.generate_monolithic_code()
		f = open(f'{self.kernel_name}.c', 'w')
		f.write(output)
		f.close()

def linear_strain(gradu):
	return (gradu + gradu.T) / 2

##################################################################
##################################################################

class AT2:
	def g(self, c):
		return (1 - c)**2

	def omega(self, c):
		return c**2

	def comega(self, c):
	 	return 2

class IsotropicPhaseField(PhaseFieldBase):
	def __init__(self, name, AT, elem_trial, elem_test):
		super().__init__(elem_trial, elem_test)

		mu, lmbda = sp.symbols('mu lambda', real=True)
		Gc, ls = sp.symbols('Gc ls', real=True)

		self.kernel_name = name
		self.params = [mu, lmbda, Gc, ls]

		gradu = self.displacement.grad()
		gradc = self.phase.grad()
		c = self.phase.value()

		# Energy
		epsu = linear_strain(gradu)
		eu = lmbda/2 * tr(epsu) * tr(epsu) + mu * inner(epsu, epsu)
		ec = (Gc / AT.comega(c)) * (AT.omega(c)/ls + ls * inner(gradc, gradc))
		energy = AT.g(c) * eu + ec
		
		self.initialize(energy)

# pp2 = IsotropicPhaseField("IsotropicPhaseField_2D_AT2", AT2(), GenericFE('trial', 2), GenericFE('test', 2))
# pp2.genreate_split_code()
# pp2.generate_code()

pp3 = IsotropicPhaseField("IsotropicPhaseField_3D_AT2", AT2(), GenericFE('trial', 3), GenericFE('test', 3))
pp3.generate_code()
pp3.genreate_split_code()

############################################################# 

