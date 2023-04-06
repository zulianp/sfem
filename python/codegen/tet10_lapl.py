#!/usr/bin/env python3

from sfem_codegen import *
from tet10 import Tet10
from laplace_op import LaplaceOp

simplify_expr = False

fe = Tet10()
op = LaplaceOp(fe, [qx, qy, qz])

print_hessian = False
print_gradient = False
print_value = False

print_hessian = True
print_gradient = True
print_value = True

if print_hessian:
	print('Hessian')
	print('---------------------------------------------------')
	c_code(op.hessian())
	print('---------------------------------------------------')

if print_gradient:
	print('Gradient')
	print('---------------------------------------------------')
	c_code(op.gradient())
	print('---------------------------------------------------')

if print_value:
	print('Value')
	print('---------------------------------------------------')
	c_code(op.value())
	print('---------------------------------------------------')
	

# dims = fe.n_dims()
# qp = sp.Matrix(dims, 1, [qx, qy, qz])
# f = fe.fun(qp)
# g = fe.grad(qp)

# u = coeffs('u', fe.n_nodes())

# affine_dV = det3(A) / 6

# FFF = affine_dV * (Ainv * Ainv.T)
# cFFF = sp.Matrix(dims, dims, [0]*(dims*dims))

# FFF_x_g = sp.Matrix(dims, 1, [0] * dims)
# for i in range(0, fe.n_nodes()):
# 	gi = cFFF * g[i]
# 	for d in range(0, dims):
# 		FFF_x_g[d] += gi[d] * u[i]


# c_jac_inv = sp.Matrix(dims, dims, coeffs('c_jac_inv', dims*dims))

# grad_uh = sp.Matrix(dims, 1, [0] * dims)
# for i in range(0, fe.n_nodes()):
# 	gi = g[i]
# 	for d in range(0, dims):
# 		grad_uh[d] += gi[d] * u[i]
# grad_uh = c_jac_inv * grad_uh


# # Sub-parametric
# varidx = 0
# for i in range(0, dims):
# 	for j in range(i, dims):
# 		var = sp.symbols(f'fff[{varidx}*stride]')
# 		varidx += 1
# 		cFFF[i, j] = var
# 		cFFF[j, i] = var;



# # print_hessian = True
# # print_gradient = True
# print_value = True

# if print_hessian:
# 	expr = []
# 	for i in range(0, fe.n_nodes()):
# 		gi = cFFF * g[i]

# 		for j in range(0, fe.n_nodes()):
# 			integr = 0
# 			for d in range(0, dims):
# 				gdotg = gi[d] * g[j][d]
# 				integr += sp.integrate(gdotg, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) 

# 			var = sp.symbols(f'element_hessian[{i*fe.n_nodes() + j}*stride]')
# 			expr.append(ast.Assignment(var, integr))


# 	print('Matrix')
# 	print('---------------------------------------------------')
# 	c_code(expr)
# 	print('---------------------------------------------------')

# if print_gradient:
# 	expr = []
# 	for i in range(0, fe.n_nodes()):
# 		integr = 0

# 		for d in range(0, dims):
# 			gdotg = FFF_x_g[d] * g[i][d]
# 			integr += sp.integrate(gdotg, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) 

# 		lform = sp.symbols(f'element_vector[{i}*stride]')
# 		expr.append(ast.Assignment(lform, integr))

# 	print('---------------------------------------------------')
# 	c_code(expr)
# 	print('---------------------------------------------------')

# if print_value:
# 	expr = []
# 	integr = 0

# 	for d in range(0, fe.n_dims()):
# 		gsquared = sp.integrate((grad_uh[d] **2)/2 , (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) 
# 		integr += gsquared


# 	for d1 in range(0, fe.n_dims()):
# 		for d2 in range(0, fe.n_dims()):
# 			integr = integr.subs(c_jac_inv[d1, d2], Ainv[d1, d2])

# 	form = sp.symbols(f'element_scalar[0]')
# 	expr.append(ast.Assignment(form, integr))

# 	print('---------------------------------------------------')
# 	c_code(expr)
# 	print('---------------------------------------------------')
