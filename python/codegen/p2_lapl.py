#!/usr/bin/env python3

from sfem_codegen import *

simplify_expr = False

class FE:
	# def __init__(self):

	def grad(self, p):
		fx = self.fun(p)
		dims, __ = p.shape
		nn = len(fx)

		g = [0] * nn

		for i in range(0, nn):
			gi = []

			for d in range(0, dims):
				gi.append(sp.diff(fx[i], p[d]))

			g[i] = sp.Matrix(dims, 1, gi)
		return g

class Tet10(FE):
	def __init__(self):
		super().__init__()

	def f0(self, x, y, z):
		l0 = 1 - x - y - z
		return (2 * l0 - 1) * l0

	def f1(self, x, y, z):
	    l1 = x
	    return (2 * l1 - 1) * l1

	def f2(self, x, y, z):
	    l2 = y
	    return (2 * l2 - 1) * l2

	def f3(self, x, y, z):
	    l3 = z
	    return (2 * l3 - 1) * l3

	def f4(self, x, y, z):
	    l0 = 1 - x - y - z
	    l1 = x
	    return 4 * l0 * l1

	def f5(self, x, y, z):
	    l1 = x
	    l2 = y
	    return 4 * l1 * l2

	def f6(self, x, y, z):
	    l0 = 1 - x - y - z
	    l2 = y
	    return 4 * l0 * l2

	def f7(self, x, y, z):
	    l0 = 1 - x - y - z
	    l3 = z
	    return 4 * l0 * l3

	def f8(self, x, y, z):
	    l1 = x
	    l3 = z
	    return 4 * l1 * l3

	def f9(self, x, y, z):
	    l2 = y
	    l3 = z
	    return 4 * l2 * l3

	def fun(self, p):
		x = p[0]
		y = p[1]
		z = p[2]

		return [
			self.f0(x, y, z),
			self.f1(x, y, z),
			self.f2(x, y, z),
			self.f3(x, y, z),
			self.f4(x, y, z),
			self.f5(x, y, z),
			self.f6(x, y, z),
			self.f7(x, y, z),
			self.f8(x, y, z),
			self.f9(x, y, z)
		]

	def n_nodes(self):
		return 10

	def n_dims(self):
		return 3

fe = Tet10()

dims = fe.n_dims()
qp = sp.Matrix(dims, 1, [qx, qy, qz])
f = fe.fun(qp)
g = fe.grad(qp)

u = coeffs('u', fe.n_nodes())

affine_dV = det3(A) / 6

FFF = affine_dV * (Ainv * Ainv.T)
cFFF = sp.Matrix(dims, dims, [0]*(dims*dims))

FFF_x_g = sp.Matrix(dims, 1, [0] * dims)
for i in range(0, fe.n_nodes()):
	gi = cFFF * g[i]
	for d in range(0, dims):
		FFF_x_g[d] += gi[d] * u[i]


c_jac_inv = sp.Matrix(dims, dims, coeffs('c_jac_inv', dims*dims))

grad_uh = sp.Matrix(dims, 1, [0] * dims)
for i in range(0, fe.n_nodes()):
	gi = g[i]
	for d in range(0, dims):
		grad_uh[d] += gi[d] * u[i]
grad_uh = c_jac_inv * grad_uh


# Sub-parametric
varidx = 0
for i in range(0, dims):
	for j in range(i, dims):
		var = sp.symbols(f'fff[{varidx}*stride]')
		varidx += 1
		cFFF[i, j] = var
		cFFF[j, i] = var;


print_matrix = False
print_gradient = False
print_value = False

# print_matrix = True
# print_gradient = True
print_value = True

if print_matrix:
	expr = []
	for i in range(0, fe.n_nodes()):
		gi = cFFF * g[i]

		for j in range(0, fe.n_nodes()):
			integr = 0
			for d in range(0, dims):
				gdotg = gi[d] * g[j][d]
				integr += sp.integrate(gdotg, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) 

			var = sp.symbols(f'element_matrix[{i*fe.n_nodes() + j}*stride]')
			expr.append(ast.Assignment(var, integr))


	print('Matrix')
	print('---------------------------------------------------')
	c_code(expr)
	print('---------------------------------------------------')

if print_gradient:
	expr = []
	for i in range(0, fe.n_nodes()):
		integr = 0

		for d in range(0, dims):
			gdotg = FFF_x_g[d] * g[i][d]
			integr += sp.integrate(gdotg, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) 

		lform = sp.symbols(f'element_vector[{i}*stride]')
		expr.append(ast.Assignment(lform, integr))

	print('---------------------------------------------------')
	c_code(expr)
	print('---------------------------------------------------')

if print_value:
	expr = []
	integr = 0

	for d in range(0, fe.n_dims()):
		gsquared = sp.integrate((grad_uh[d] **2)/2 , (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) 
		integr += gsquared


	for d1 in range(0, fe.n_dims()):
		for d2 in range(0, fe.n_dims()):
			integr = integr.subs(c_jac_inv[d1, d2], Ainv[d1, d2])

	form = sp.symbols(f'element_scalar[0]')
	expr.append(ast.Assignment(form, integr))

	print('---------------------------------------------------')
	c_code(expr)
	print('---------------------------------------------------')
