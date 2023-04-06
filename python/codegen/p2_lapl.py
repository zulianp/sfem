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
	    return (2.0 * l1 - 1.0) * l1

	def f2(self, x, y, z):
	    l2 = y
	    return (2.0 * l2 - 1.0) * l2

	def f3(self, x, y, z):
	    l3 = z
	    return (2.0 * l3 - 1.0) * l3

	def f4(self, x, y, z):
	    l0 = 1 - x - y - z
	    l1 = x
	    return 4.0 * l0 * l1

	def f5(self, x, y, z):
	    l1 = x
	    l2 = y
	    return 4.0 * l1 * l2

	def f6(self, x, y, z):
	    l0 = 1 - x - y - z
	    l2 = y
	    return 4.0 * l0 * l2

	def f7(self, x, y, z):
	    l0 = 1 - x - y - z
	    l3 = z
	    return 4.0 * l0 * l3

	def f8(self, x, y, z):
	    l1 = x
	    l3 = z
	    return 4.0 * l1 * l3

	def f9(self, x, y, z):
	    l2 = y
	    l3 = z
	    return 4.0 * l2 * l3

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

fe = Tet10()

dims = 3
qp = sp.Matrix(dims, 1, [qx, qy, qz])
f = fe.fun(qp)
g = fe.grad(qp)
print(g)

affine_dV = det3(A) / 6

FFF = affine_dV * (Ainv * Ainv.T)
cFFF = sp.Matrix(dims, dims, [0]*(dims*dims))

varidx = 0
for i in range(0, dims):
	for j in range(i, dims):
		var = sp.symbols(f'jac_inv[{varidx}*stride]')
		varidx += 1
		cFFF[i, j] = var
		cFFF[j, i] = var;

# Sub-parametric

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
		# print(integr)

print('---------------------------------------------------')
c_code(expr)
print('---------------------------------------------------')

# listu = []
# for i in range(0, 4):
# 	ui= sp.symbols(f'u[{i}]', real=True)
# 	listu.append(ui)

# u = sp.Matrix(4, 1, listu)
# grad_uh = sp.Matrix(3, 1, [0, 0, 0])

# for i in range(0, 4):
# 	for d in range(0, 3):
# 		grad_uh[d] += sp.diff(f[i], q[d]) * u[i]

# if False:
# # if True:
# 	expr = []
# 	for i in range(0, 4):
# 		for j in range(0, 4):
# 			integr = 0
# 			for d in range(0, 3):
# 				integr += sp.diff(f[i], q[d]) * sp.diff(f[j], q[d]) * dV
# 			bform = sp.symbols(f'element_matrix[{i*4+j}]')

# 			if simplify_expr:
# 				integr = sp.simplify(integr)

# 			expr.append(ast.Assignment(bform, integr))
# 	print('---------------------------------------------------')
# 	c_code(expr)
# 	print('---------------------------------------------------')

# # if False:
# if True:
# 	expr = []
# 	for i in range(0, 4):
# 		integr = 0

# 		for d in range(0, 3):
# 			integr += grad_uh[d] * sp.diff(f[i], q[d])

# 		integr = integr * dV

# 		lform = sp.symbols(f'element_vector[{i}]')

# 		if simplify_expr:
# 			integr = sp.simplify(integr)

# 		expr.append(ast.Assignment(lform, integr))

# 	print('---------------------------------------------------')
# 	c_code(expr)
# 	print('---------------------------------------------------')

# if False:
# # if True:
# 	expr = []
# 	integr = 0
# 	for d in range(0, 3):
# 		integr += (grad_uh[d] **2)/2 

# 	integr *= dV

# 	form = sp.symbols(f'element_scalar[0]')

# 	if simplify_expr:
# 		integr = sp.simplify(integr)

# 	expr.append(ast.Assignment(form, integr))

# 	print('---------------------------------------------------')
# 	c_code(expr)
# 	print('---------------------------------------------------')
