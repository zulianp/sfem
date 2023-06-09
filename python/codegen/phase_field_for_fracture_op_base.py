from sfem_codegen import *
from fe_material import FEMaterial
from fe_material import FEFunction

from time import perf_counter

# import pdb

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
		subs = [(s_disp_grad, e_disp_grad), (s_gradc, e_gradc), (stot(s_c), stot(e_c)), (s_jac_inv, e_jac_inv)]

		integr_value = self.subs(subs, fe.integrate(q, e))
		# print(integr_value)

		integr_gradient = self.subs_tensors(subs, self.integrate(fe, q, eval_grad))
		integr_hessian = self.subs_tensors(subs, self.integrate(fe, q, eval_hessian))

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
	
	def get_eval_hessian(self):
		return self.integr_hessian

	def get_eval_gradient(self):
		return self.integr_gradient

	def get_eval_value(self):
		return self.integr_value 
