from sfem_codegen import *
from fe_material import FEMaterial
from fe_material import FEFunction

from time import perf_counter

import pdb

class PhaseFieldForFractureOpBase(FEMaterial):
	def energy(self, c, gradc, gradu):
		print("Implement Me!")
		assert False 

	def __init__(self, fe):
		self.fe = fe
		self.initialize()

	def initialize(self):
		fe = self.fe
		fe_disp = FEFunction("disp", fe, fe.manifold_dim())
		fe_phase = FEFunction("phase", fe, 1)

		self.fe_disp = fe_disp
		self.fe_phase = fe_phase

		s_disp_grad = fe_disp.grad().var()
		s_c = fe_phase.value().var()[0]
		s_gradc = fe_phase.grad().var()

		e_disp_grad = fe_disp.grad().expansion()
		e_c = fe_phase.value().expansion()
		e_gradc = fe_phase.grad().expansion()

		###################################################################
		# Material law
		###################################################################
		c_log("Material law")

		# strain energy function
		e = self.energy(s_c, s_gradc, s_disp_grad)

		nnodes = fe.n_nodes()
		ndofs = nnodes * (fe.manifold_dim() + 1)

		###################################################################
		# Gradient
		###################################################################
		c_log("Compute derivatives")

		eval_grad = sp.Matrix(ndofs, 1, [0] * ndofs)

		# Phase-field
		dedc = sp.diff(e, s_c)
		dedgradc = sp.diff(e, s_gradc)
		eval_grad_c = self.mult_list(dedc, fe_phase.shape_fun()) 
		eval_grad_c += self.inner_list(dedgradc, fe_phase.shape_grad())
		eval_grad[0:nnodes,:] = eval_grad_c[:,:]

		# Displacement
		dedu = self.derivative(e, s_disp_grad)
		eval_grad_u = self.inner_list(dedu, fe_disp.shape_grad())
		eval_grad[nnodes:ndofs,:] = eval_grad_u[:,:]

		###################################################################
		# Hessian
		###################################################################
		eval_hessian = sp.Matrix(ndofs, ndofs, [0] * (ndofs * ndofs))

		# Phase-field
		eval_hessian_c = self.vector_diff_scalar_x(eval_grad, s_c, fe_phase.shape_fun()) 
		eval_hessian_c += self.vector_diff_tensor_x(eval_grad, s_gradc, fe_phase.shape_grad()) 
		eval_hessian[0:nnodes,0:ndofs] = eval_hessian_c[:,:]

		# Displacement
		eval_hessian[nnodes:ndofs,0:ndofs] = self.vector_diff_tensor_x(eval_grad, s_disp_grad, fe_disp.shape_grad()) 

		###################################################################
		# Integrate and substitute
		###################################################################
		c_log("Integrate")

		full_eval = True
		q = fe_disp.quadrature_point()
		s_jac_inv = fe.symbol_jacobian_inverse()
		e_jac_inv = fe.jacobian_inverse(q)
		# print(e_jac_inv)

		dV = fe.jacobian_determinant(q)
		pres_subs = [(s_disp_grad, e_disp_grad), (s_gradc, e_gradc), (stot(s_c), stot(e_c))]
		post_subs = [(s_jac_inv, e_jac_inv)]

		e = self.subs(pres_subs, e)
		integr_value = self.subs(post_subs, fe.integrate(q, e))

		eval_grad = self.subs_tensors(pres_subs, eval_grad)
		integr_gradient = self.subs_tensors(post_subs, self.integrate(fe, q, eval_grad))

		eval_hessian = self.subs_tensors(pres_subs, eval_hessian)
		integr_hessian = self.subs_tensors(post_subs, self.integrate(fe, q, eval_hessian))

		# Assume constant dV for efficiency reasons
		integr_value *= dV
		integr_gradient = self.scale_tensors(dV, integr_gradient)
		integr_hessian  = self.scale_tensors(dV, integr_hessian)
		###################################################################

		self.e = e
		self.disp = fe_disp.coeffs()
		self.c = fe_phase.coeffs()

		self.eval_grad = eval_grad
		self.eval_hessian = eval_hessian

		self.integr_value = integr_value
		self.integr_gradient = integr_gradient
		self.integr_hessian = integr_hessian

		###################################################################
		self.fe = fe
		###################################################################

	def get_eval_gradient_u(self):
		nnodes = self.fe.n_nodes()
		ndofs = nnodes * (self.fe.manifold_dim() + 1)
		return self.integr_gradient[nnodes:ndofs,:]

	def get_eval_gradient_c(self):
		nnodes = self.fe.n_nodes()
		return self.integr_gradient[0:nnodes,:]

	def gradient_u(self):
		return self.assign_vector(self.get_eval_gradient_u())

	def gradient_c(self):
		return self.assign_vector(self.get_eval_gradient_c())

	def get_eval_hessian_uu(self):
		nnodes = self.fe.n_nodes()
		ndofs = nnodes * (self.fe.manifold_dim() + 1)
		return self.integr_hessian[nnodes:ndofs, nnodes:ndofs]

	def get_eval_hessian_uc(self):
		nnodes = self.fe.n_nodes()
		ndofs = nnodes * (self.fe.manifold_dim() + 1)
		return self.integr_hessian[nnodes:ndofs,0:nnodes]

	def get_eval_hessian_cu(self):
		nnodes = self.fe.n_nodes()
		ndofs = nnodes * (self.fe.manifold_dim() + 1)
		return self.integr_hessian[0:nnodes,nnodes:ndofs]

	def get_eval_hessian_uc(self):
		nnodes = self.fe.n_nodes()
		ndofs = nnodes * (self.fe.manifold_dim() + 1)
		return self.integr_hessian[nnodes:ndofs,0:nnodes]

	def hessian_uu(self):
		return self.assign_matrix(self.get_eval_hessian_uu())

	def hessian_uc(self):
		return self.assign_matrix(self.get_eval_hessian_uc())

	def hessian_cu(self):
		return self.assign_matrix(self.get_eval_hessian_cu())

	def hessian_cc(self):
		return self.assign_matrix(self.get_eval_hessian_cc())

	def apply_matrix(self, mat):
		rows, cols = mat.shape
		self.increment = coeffs('increment', cols)
		inc = self.increment

		expr = []
		for i in range(0, rows):
			val = 0
			for j in range(0, cols):
				val += mat[i, j] * inc[j]

			var = sp.symbols(f'element_vector[{i}*stride]')
			expr.append(ast.Assignment(var, val))
		return expr

	def apply_uu(self):
		return self.apply_matrix(self.get_eval_hessian_uu())

	def apply_cc(self):
		return self.apply_matrix(self.get_eval_hessian_cc())

	def apply_uc(self):
		return self.apply_matrix(self.get_eval_hessian_uc())

	def apply_cu(self):
		return self.apply_matrix(self.get_eval_hessian_cu())

	def get_eval_hessian_cc(self):
		nnodes = self.fe.n_nodes()
		return self.integr_hessian[0:nnodes,0:nnodes]
	
	def get_eval_hessian(self):
		return self.integr_hessian

	def get_eval_gradient(self):
		return self.integr_gradient

	def get_eval_value(self):
		return self.integr_value 

	def generate_c_code(self):
		# self.read_tpl()
		
		# tpl_mono  = self.tpl_mono
		# tpl_split = self.tpl_split

		material_name = 'phase_field_for_fracture'
		singature_prefix = f'SFEM_INLINE static void {material_name}'

		param_list = ""
		param_list += f'// material parameters \n'
		for p in self.params:
			param_list += f'const {real_t} {p},\n'

		coord_names = ['x', 'y', 'z', 't']
		coords = self.fe.coords_sub_parametric()
		for d in range(0, self.fe.spatial_dim()):
			param_list += f'// {coord_names[d]} coordinates\n'
			for c in coords[d]:
				param_list += f'const {real_t} {c},\n'

		param_list += f'// data arrays \n'
		param_list += f'const {real_t} * SFEM_RESTRICT {self.fe_phase.name()},\n'

		# param_grad_phase = f'const {real_t} * SFEM_RESTRICT grad_{self.fe_phase.name()},\n'
		param_disp  = f'const {real_t} * SFEM_RESTRICT {self.fe_disp.name()},\n'
		param_inc  = f'const {real_t} * SFEM_RESTRICT increment,\n'


		output = f"// Automatically generate code for {material_name}\n"

		includes = "#include \"sfem_base.h\"\n"
		includes += "#include \"sfem_vec.h\"\n"
		includes += "#include \"math.h\"\n"
		output += includes

		output += "static const int stride = 1;\n"

		if True:
			output += "\n"
			# Value kernel
			value_signature = singature_prefix + f'_value(\n' + param_list + param_disp
			value_signature += f'{real_t} * SFEM_RESTRICT element_scalar\n' + ')\n'

			value_body = c_gen(self.value())
			value_kernel = value_signature + '{\n' + f'{value_body}' + '\n}\n'

			output += value_kernel

		if True:
			output += "\n"
			gradient_signature = singature_prefix + f'_gradient(\n' + param_list + param_disp
			gradient_signature += f'{real_t} * SFEM_RESTRICT element_vector\n' + ')\n'

			gradient_body = c_gen(self.gradient())
			gradient_kernel = gradient_signature + '{\n' + f'{gradient_body}' + '\n}\n'

			output += gradient_kernel

		if True:
			output += "\n"
			hessian_signature = singature_prefix + f'_hessian(\n' + param_list + param_disp
			hessian_signature += f'{real_t} * SFEM_RESTRICT element_matrix\n' + ')\n'

			hessian_body = c_gen(self.hessian())
			hessian_kernel = hessian_signature + '{\n' + f'{hessian_body}' + '\n}\n'

			output += hessian_kernel
		
		output += "// Split evaliations\n"

		if True:
			output += "\n"
			gradient_u_signature = singature_prefix + f'_gradient_u(\n' + param_list + param_disp
			gradient_u_signature += f'{real_t} * SFEM_RESTRICT element_vector\n' + ')\n'

			gradient_u_body = c_gen(self.gradient_u())
			gradient_u_kernel = gradient_u_signature + '{\n' + f'{gradient_u_body}' + '\n}\n'

			output += gradient_u_kernel

		if True:
			output += "\n"
			gradient_c_signature = singature_prefix + f'_gradient_c(\n' + param_list + param_disp
			gradient_c_signature += f'{real_t} * SFEM_RESTRICT element_vector\n' + ')\n'

			gradient_c_body = c_gen(self.gradient_c())
			gradient_c_kernel = gradient_c_signature + '{\n' + f'{gradient_c_body}' + '\n}\n'

			output += gradient_c_kernel

		if True:
			output += "\n"
			hessian_cc_signature = singature_prefix + f'_hessian_cc(\n' + param_list + param_disp
			hessian_cc_signature += f'{real_t} * SFEM_RESTRICT element_matrix\n' + ')\n'

			hessian_cc_body = c_gen(self.hessian_cc())
			hessian_cc_kernel = hessian_cc_signature + '{\n' + f'{hessian_cc_body}' + '\n}\n'

			output += hessian_cc_kernel

		if True:
			output += "\n"
			hessian_uu_signature = singature_prefix + f'_hessian_uu(\n' + param_list + param_disp
			hessian_uu_signature += f'{real_t} * SFEM_RESTRICT element_matrix\n' + ')\n'

			hessian_uu_body = c_gen(self.hessian_uu())
			hessian_uu_kernel = hessian_uu_signature + '{\n' + f'{hessian_uu_body}' + '\n}\n'

			output += hessian_uu_kernel

		if True:
			output += "\n"
			hessian_uc_signature = singature_prefix + f'_hessian_uc(\n' + param_list + param_disp
			hessian_uc_signature += f'{real_t} * SFEM_RESTRICT element_matrix\n' + ')\n'

			hessian_uc_body = c_gen(self.hessian_uc())
			hessian_uc_kernel = hessian_uc_signature + '{\n' + f'{hessian_uc_body}' + '\n}\n'

			output += hessian_uc_kernel

		if True:
			output += "\n"
			hessian_cu_signature = singature_prefix + f'_hessian_cu(\n' + param_list + param_disp
			hessian_cu_signature += f'{real_t} * SFEM_RESTRICT element_matrix\n' + ')\n'

			hessian_cu_body = c_gen(self.hessian_cu())
			hessian_cu_kernel = hessian_cu_signature + '{\n' + f'{hessian_cu_body}' + '\n}\n'

			output += hessian_cu_kernel

		if True:
			output += "\n"
			apply_signature = singature_prefix + f'_apply(\n' + param_list + param_disp + param_inc
			apply_signature += f'{real_t} * SFEM_RESTRICT element_vector\n' + ')\n'

			apply_body = c_gen(self.apply())
			apply_kernel = apply_signature + '{\n' + f'{apply_body}' + '\n}\n'

			output += apply_kernel

		if True:
			output += "\n"
			apply_uu_signature = singature_prefix + f'_apply_uu(\n' + param_list + param_disp + param_inc
			apply_uu_signature += f'{real_t} * SFEM_RESTRICT element_vector\n' + ')\n'

			apply_uu_body = c_gen(self.apply_uu())
			apply_uu_kernel = apply_uu_signature + '{\n' + f'{apply_uu_body}' + '\n}\n'

			output += apply_uu_kernel

		if True:
			output += "\n"
			apply_cc_signature = singature_prefix + f'_apply_cc(\n' + param_list + param_disp + param_inc
			apply_cc_signature += f'{real_t} * SFEM_RESTRICT element_vector\n' + ')\n'

			apply_cc_body = c_gen(self.apply_cc())
			apply_cc_kernel = apply_cc_signature + '{\n' + f'{apply_cc_body}' + '\n}\n'

			output += apply_cc_kernel

		if True:
			output += "\n"
			apply_uc_signature = singature_prefix + f'_apply_uc(\n' + param_list + param_disp + param_inc
			apply_uc_signature += f'{real_t} * SFEM_RESTRICT element_vector\n' + ')\n'

			apply_uc_body = c_gen(self.apply_uc())
			apply_uc_kernel = apply_uc_signature + '{\n' + f'{apply_uc_body}' + '\n}\n'

			output += apply_uc_kernel

		if True:
			output += "\n"
			apply_cu_signature = singature_prefix + f'_apply_cu(\n' + param_list + param_disp + param_inc
			apply_cu_signature += f'{real_t} * SFEM_RESTRICT element_vector\n' + ')\n'

			apply_cu_body = c_gen(self.apply_cu())
			apply_cu_kernel = apply_cu_signature + '{\n' + f'{apply_cu_body}' + '\n}\n'

			output += apply_cu_kernel

		with open(f'{self.fe.name()}_{material_name}_kernels.c', 'w') as f:
			f.write(output)
			f.close()

	def read_file(self, path):
		with open(path, 'r') as f:
		    tpl = f.read()
		    return tpl
		assert False
		return ""

	def read_tpl(self):
		self.tpl_mono  = self.read_file('tpl/PhaseField_tpl.c')
		self.tpl_split = self.read_file('tpl/PhaseFieldSplit_tpl.c')

