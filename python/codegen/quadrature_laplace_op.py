#!/usr/bin/env python3

from sfem_codegen import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *

# import pdb

def read_file(path):
	with open(path, 'r') as f:
	    tpl = f.read()
	    return tpl
	assert False
	return ""

def str_to_file(path, mystr):
	with open(path, 'w') as f:
		f.write(mystr)
		f.close()

def assign_fff(name, mat):
	rows, cols = mat.shape

	expr = []
	idx = 0
	for i in range(0, rows):
		for j in range(i, cols):
			var = sp.symbols(f'{name}[{idx}]')
			expr.append(ast.Assignment(var, mat[i, j]))
			idx += 1
	return expr

def assign_matrix(name, mat):
	rows, cols = mat.shape

	expr = []
	for i in range(0, rows):
		for j in range(0, cols):
			var = sp.symbols(f'{name}[{i*cols + j}]')
			expr.append(ast.Assignment(var, mat[i, j]))
	return expr

class QuadratureLaplaceOp:
	def __init__(self, fe, q):

		dims = fe.manifold_dim()
		q = sp.Matrix(dims, 1, q)

		f = fe.fun(q)
		g = fe.grad(q)

		u = coeffs('u', fe.n_nodes())

		jac_inv = fe.jacobian_inverse(q)
		dV = fe.jacobian_determinant(q) * fe.reference_measure()
		FFF = (jac_inv * jac_inv.T) * dV 

		###################################################################

		cFFF = sp.Matrix(dims, dims, [0]*(dims*dims))
		varidx = 0
		for i in range(0, dims):
			for j in range(i, dims):
				var = sp.symbols(f'fff[{varidx}*stride]')
				varidx += 1
				cFFF[i, j] = var
				cFFF[j, i] = var;

		###################################################################

		FFF_x_g = sp.Matrix(dims, 1, [0] * dims)
		for i in range(0, fe.n_nodes()):
			for d in range(0, dims):
				FFF_x_g[d] += g[i][d] * u[i]


		qw = sp.symbols('qw')
		FFF_x_g = cFFF * FFF_x_g * qw
		################################################################### 
		c_jac_inv = sp.Matrix(dims, dims, coeffs('c_jac_inv', dims*dims))

		grad_uh = sp.Matrix(dims, 1, [0] * dims)
		for i in range(0, fe.n_nodes()):
			gi = g[i]

			for d in range(0, dims):
				grad_uh[d] += gi[d] * u[i]

		grad_uh = c_jac_inv * grad_uh


		gref = [0]*dims
		for d in range(0, dims):
			gref[d] = sp.zeros(fe.n_nodes(), 1)
			for i in range(0, fe.n_nodes()):
				gref[d][i] = g[i][d]
			
		print(gref[0])


		print(gref[1])
		

		print(gref[2])
		
		tpl = read_file('tpl/quadrature_laplace_op_tpl.c')
		code = tpl.format(
			TRIAL_OPERAND_CODE=c_gen(assign_matrix("out", FFF_x_g)),
			REF_SHAPE_GRAD_X_CODE=c_gen(assign_matrix("out",  gref[0])),
			REF_SHAPE_GRAD_Y_CODE=c_gen(assign_matrix("out",  gref[1])),
			REF_SHAPE_GRAD_Z_CODE=c_gen(assign_matrix("out",  gref[2])))

		c_log(code)

		###################################################################

		self.fe = fe
		self.f = f
		self.g = g
		self.u = u
		self.q = q

		self.grad_uh = grad_uh

		self.dV = dV
		self.c_jac_inv = c_jac_inv

		self.FFF_x_g = FFF_x_g
		self.cFFF = cFFF
		self.FFF = FFF
		self.jac_inv = jac_inv

	def fff(self):
		expr = []

		for d1 in range(0, self.fe.spatial_dim()):
			for d2 in range(d1,  self.fe.spatial_dim()):
				var = self.cFFF[d1, d2]
				val = self.FFF[d1, d2]
				expr.append(ast.Assignment(var, val))

		return expr

	def hessian(self):
		fe = self.fe
		g = self.g
		cFFF = self.cFFF
		fe = self.fe
		q = self.q

		expr = []
		for i in range(0, fe.n_nodes()):
			gi = cFFF * g[i]

			for j in range(0, fe.n_nodes()):
				integr = 0
				for d in range(0, fe.manifold_dim()):
					gdotg = gi[d] * g[j][d]
					integr += fe.integrate(q, gdotg)

				var = sp.symbols(f'element_matrix[{i*fe.n_nodes() + j}*stride]')
				expr.append(ast.Assignment(var, integr))

		return expr

	def gradient(self):
		fe = self.fe
		g = self.g
		FFF_x_g = self.FFF_x_g
		q = self.q

		expr = []
		for i in range(0, fe.n_nodes()):
			integr = 0

			for d in range(0, fe.manifold_dim()):
				gdotg = FFF_x_g[d] * g[i][d]
				integr += fe.integrate(q, gdotg)

			lform = sp.symbols(f'element_vector[{i}*stride]')
			expr.append(ast.Assignment(lform, integr))

		# pdb.set_trace()
		return expr

	def value(self):
		fe = self.fe
		integr = 0
		q = self.q
		c_jac_inv = self.c_jac_inv
		grad_uh = self.grad_uh
		jac_inv = self.jac_inv
		dV = self.dV

		expr = []
		for d in range(0, fe.manifold_dim()):
			gsquared = fe.integrate(q, (grad_uh[d] **2)) / 2
			integr += gsquared


		for d1 in range(0, fe.manifold_dim()):
			for d2 in range(0, fe.manifold_dim()):
				integr = integr.subs(c_jac_inv[d1, d2], jac_inv[d1, d2])

		integr *= dV

		form = sp.symbols(f'element_scalar[0]')
		expr.append(ast.Assignment(form, integr))
		return expr

def main():
	# fe = Tri6()
	# fe = Tri3()
	# q = sp.Matrix(2, 1, [qx, qy])
	# op = QuadratureLaplaceOp(fe, q)

	# fe = Tet4()
	fe = Tet10()

	q = sp.Matrix(3, 1, [qx, qy, qz])
	op = QuadratureLaplaceOp(fe, q)

	print("FFF")
	c_code(op.fff())

	# print("Hessian")
	# c_code(op.hessian())

	# print("Gradient")
	# c_code(op.gradient())

if __name__ == '__main__':
	main()