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

		d_trial_grad = self.displacement.elem_type.grad()
		d_test_grad = tp_test.grad()
		n = len(d_test_grad)
		assert(n == tp_test.block_size)

		dedc = derivative(energy, phase.value())
		dedgradc = derivative(energy, phase.grad())

		# Gradient
		self.grad_wrt_c = (dedc * test.value()) + inner(dedgradc, test.grad())

		dedu = derivative(energy, displacement.grad())
		self.grad_wrt_u = sp.Matrix(n, 1, [0]*n)

		for d1 in range(0, n):
			self.grad_wrt_u[d1] = inner(dedu, d_test_grad[d1])	

		# Hessian
		self.hessian_wrt_uu = sp.Matrix(n, n, [0]*(n*n))

		H_grad_u = sp.Matrix(n, 1, [0]*n)
		for d1 in range(0, n):
			H_grad_u[d1] = inner(dedu, d_trial_grad[d1])

		for i_trial in range(0, n):
			d2edu2 = derivative(H_grad_u[i_trial], displacement.grad())

			for i_test in range(0, n):
				self.hessian_wrt_uu[i_trial, i_test] = inner(d2edu2, d_test_grad[i_test])

		self.hessian_wrt_uc = sp.Matrix(n, 1, [0] * n)

		for i_trial in range(0, n):
			d2edudc = derivative(H_grad_u[i_trial], phase.grad())
			self.hessian_wrt_uc[i_trial] = inner(d2edudc, test.grad()) + sp.diff(H_grad_u[i_trial], phase.value()) * test.value()
		
		# cc
		grad_wrt_c = inner(dedgradc, trial.grad()) + dedc * trial.value()
		self.hessian_wrt_cc = inner(derivative(grad_wrt_c, phase.grad()), test.grad()) + derivative(grad_wrt_c, phase.value()) * test.value()

	def makegradu(self, idx, i):
		integr = self.grad_wrt_u[i] * self.dV
		
		if simplify_expr:
			integr = sp.simplify(integr)

		lform = sp.symbols(f'element_vector[{idx}]')
		expr = ast.AddAugmentedAssignment(lform, integr)
		return expr

	def makegradc(self, idx):
		integr = self.grad_wrt_c * self.dV
		
		if simplify_expr:
			integr = sp.simplify(integr)

		lform = sp.symbols(f'element_vector[{idx}]')
		expr = ast.AddAugmentedAssignment(lform, integr)
		return expr

	def makegrad(self):
		ndisp = self.tp_element_test.block_size
		n = ndisp + 1

		expr = [0] * n

		for i in range(0, ndisp):
			expr[i] = self.makegradu(i, i)

		expr[ndisp] = self.makegradc(ndisp)
		return expr

	def makehessianuu(self, idx, i, j):
		n = self.tp_element_test.block_size + 1

		integr = self.hessian_wrt_uu[i, j] * self.dV

		if simplify_expr:
			integr = sp.simplify(integr)

		# bform = sp.symbols(f'element_matrix[{i * n + j}]')
		bform = sp.symbols(f'element_matrix[{idx}]')
		return ast.AddAugmentedAssignment(bform, integr)

	def makehessianuc(self, idx, i, j):
		n = self.tp_element_test.block_size + 1

		integr = self.hessian_wrt_uc[i] * self.dV

		if simplify_expr:
			integr = sp.simplify(integr)

		# bform = sp.symbols(f'element_matrix[{i * n + j}]')
		bform = sp.symbols(f'element_matrix[{idx}]')
		return ast.AddAugmentedAssignment(bform, integr)

	def makehessiancu(self, idx, j, i):
		n = self.tp_element_test.block_size + 1

		integr = self.hessian_wrt_uc[i] * self.dV

		if simplify_expr:
			integr = sp.simplify(integr)

		# bform = sp.symbols(f'element_matrix[{j * n + i}]')
		bform = sp.symbols(f'element_matrix[{idx}]')
		return ast.AddAugmentedAssignment(bform, integr)

	def makehessiancc(self, idx):
		n = self.tp_element_test.block_size + 1

		integr = self.hessian_wrt_cc * self.dV

		if simplify_expr:
			integr = sp.simplify(integr)

		# bform = sp.symbols(f'element_matrix[{i * n + i}]')
		bform = sp.symbols(f'element_matrix[{idx}]')
		return ast.AddAugmentedAssignment(bform, integr)

	def makehessian(self):
		ndisp = self.tp_element_test.block_size
		n = ndisp + 1

		expr = [0] * (n * n)

		for i in range(0, ndisp):
			expr[i * n + ndisp] = self.makehessianuc(i * n + ndisp, i, ndisp)
			expr[ndisp * n + i] = self.makehessiancu(ndisp * n + i, ndisp, i)

			for j in range(0, ndisp):
				expr[i * n + j] = self.makehessianuu(i * n + j, i, j)

		expr[ndisp * n + ndisp] = self.makehessiancc(ndisp * n + ndisp)
		return expr

	def make_split_hessian_uu(self):
		ndisp = self.tp_element_test.block_size

		expr = [0] * (ndisp * ndisp)
		for i in range(0, ndisp):
			for j in range(0, ndisp):
				expr[i * ndisp + j] = self.makehessianuu(i * ndisp + j, i, j)

		return expr

	def make_split_hessian_uc(self):
		ndisp = self.tp_element_test.block_size

		expr = [0] * ndisp
		for i in range(0, ndisp):
			expr[i] = self.makehessianuc(i, i, 0)
		return expr

	def make_split_hessian_cu(self):
		ndisp = self.tp_element_test.block_size

		expr = [0] * ndisp
		for i in range(0, ndisp):
			expr[i] = self.makehessiancu(i, 0, i)
		return expr

	def make_split_hessian_cc(self):
		expr = [self.makehessiancc(0)]
		return expr

	def genreate_split_code(self):
		energy_code = self.energy_code()
		args_e = self.energy_args()

		# Split operators
		gradient_c_expr = self.makegradc(0)	
		gradient_c_code = c_gen(gradient_c_expr)

		ndisp = self.tp_element_test.block_size
		gradient_u_expr = []
		
		for d1 in range(0, ndisp):
			gradient_u_expr.append(self.makegradu(d1, d1))

		gradient_u_code = c_gen(gradient_u_expr)

		# console.print(gradient_c_code)
		# console.print(gradient_u_code)

		hessian_uu_expr = self.make_split_hessian_uu()
		hessian_uu_code = c_gen(hessian_uu_expr)

		hessian_cc_expr = self.make_split_hessian_cc()
		hessian_cc_code = c_gen(hessian_cc_expr)

		hessian_uc_expr = self.make_split_hessian_uc()
		hessian_uc_code = c_gen(hessian_uc_expr)

		hessian_cu_expr = self.make_split_hessian_cu()
		hessian_cu_code = c_gen(hessian_cu_expr)

		params = self.param_string()
	
		args_g_u = params + f"""const real_t {self.phase.name}, 
			const real_t *{self.phase.grad_name}, 
			const real_t *{self.displacement.grad_name},
			const real_t *grad_test,
			const real_t dV,
			real_t *element_vector
			"""
		args_g_c = params + f"""const real_t {self.phase.name}, 
			const real_t *{self.phase.grad_name}, 
			const real_t *{self.displacement.grad_name},
			const real_t test,
			const real_t *grad_test,
			const real_t dV,
			real_t *element_vector
			"""
		args_H_uu = params + f"""const real_t {self.phase.name},  
			const real_t *grad_test,
			const real_t *grad_trial,
			const real_t dV,
			real_t *element_matrix"""

		args_H_uc = params + f"""const real_t {self.phase.name},  
			const real_t *{self.displacement.grad_name},
			const real_t test,
			const real_t *grad_trial,
			const real_t dV,
			real_t *element_matrix"""

		args_H_cu = args_H_uc

		args_H_cc = params + f"""const real_t *{self.displacement.grad_name},
			const real_t test,
			const real_t trial,
			const real_t *grad_test,
			const real_t *grad_trial,
			const real_t dV,
			real_t *element_matrix"""

		tpl = self.read_field_split_tpl()

		output = tpl.format(
			kernel_name = self.kernel_name,
			energy = energy_code, 
			gradient_u = gradient_u_code, 
			gradient_c = gradient_c_code, 
			hessian_uu = hessian_uu_code, 
			hessian_uc = hessian_uc_code, 
			hessian_cu = hessian_cu_code, 
			hessian_cc = hessian_cc_code, 
			args_e = args_e,
			args_g_u = args_g_u,
			args_g_c = args_g_c,
			args_H_uu = args_H_uu,
			args_H_uc = args_H_uc,
			args_H_cu = args_H_cu,
			args_H_cc = args_H_cc
		)

		return output

	def param_string(self):
		params = ""
		for p in self.params:
			params += f"const real_t {p},\n"
		return params

	def energy_code(self):
		energy_expr = self.makeenergy()
		energy_code = c_gen(energy_expr)
		return energy_code

	def energy_args(self):
		params = self.param_string()
		args_e = params + f"""const real_t {self.phase.name}, 
			const real_t *{self.phase.grad_name}, 
			const real_t *{self.displacement.grad_name},
			const real_t dV,
			real_t *element_scalar
			"""
		return args_e

	def generate_monolithic_code(self):
		energy_code = self.energy_code()
		args_e = self.energy_args()

		gradient_expr = self.makegrad()
		gradient_code = c_gen(gradient_expr)

		hessian_expr = self.makehessian()
		hessian_code = c_gen(hessian_expr)

		params = self.param_string()
		args_g = params + f"""const real_t {self.phase.name}, 
			const real_t *{self.phase.grad_name}, 
			const real_t *{self.displacement.grad_name},
			const real_t test,
			const real_t *grad_test,
			const real_t dV,
			real_t *element_vector
			"""

		args_H = params + f"""const real_t {self.phase.name},  
			const real_t *{self.displacement.grad_name},
			const real_t test,
			const real_t trial,
			const real_t *grad_test,
			const real_t *grad_trial,
			const real_t dV,
			real_t *element_matrix"""

		tpl = self.read_tpl()

		output = tpl.format(
			kernel_name = self.kernel_name,
			energy = energy_code, 
			gradient = gradient_code, 
			hessian = hessian_code, 
			args_e = args_e,
			args_g = args_g,
			args_H = args_H)

		return output

	def read_field_split_tpl(self):
		tpl_path = 'tpl/PhaseFieldSplit_tpl.c'

		tpl = None
		with open(tpl_path, 'r') as f:
		    tpl = f.read()
		    return tpl

	def read_tpl(self):
		tpl_path = 'tpl/PhaseField_tpl.c'

		tpl = None
		with open(tpl_path, 'r') as f:
		    tpl = f.read()
		    return tpl

	def generate_code(self):
		with open(f'{self.kernel_name}.c', 'w') as f:
			output = self.generate_monolithic_code()
			f.write(output)
			f.close()

		with open(f'{self.kernel_name}_split.c', 'w') as f:
			output = self.genreate_split_code()
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
	def __init__(self, name, degradation, elem_trial, elem_test):
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
		ec = (Gc / degradation.comega(c)) * (degradation.omega(c)/ls + ls * inner(gradc, gradc))
		energy = degradation.g(c) * eu + ec
		
		self.initialize(energy)

pp2 = IsotropicPhaseField("IsotropicPhaseField_2D_AT2", AT2(), GenericFE('trial', 2), GenericFE('test', 2))
pp2.generate_code()

pp3 = IsotropicPhaseField("IsotropicPhaseField_3D_AT2", AT2(), GenericFE('trial', 3), GenericFE('test', 3))
pp3.generate_code()

############################################################# 

