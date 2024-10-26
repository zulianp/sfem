#!/usr/bin/env python3

from tet10 import *
from tet4 import *
from tri6 import *
from quad4 import *
from mass_op import *
from laplace_op import *

if __name__ == '__main__':
	# fe = Quad4(False)
	# fe = Hex8(False)
	fe = Tet10()
	dim = fe.spatial_dim()

	start = perf_counter()

	c = fe.barycenter()

	point = coeffs('p', dim)
	f = fe.taylor_expand_fun(c, point)
	g = fe.taylor_expand_grad(c, point)

	n = fe.n_nodes()
	M = sp.zeros(n, n)
	L = sp.zeros(n, n)
	q = fe.quadrature_point()
	# dV = subsmat(fe.measure(q), q, c)
	dV = fe.symbol_jacobian_determinant()

	taylor_mass_matrix = []
	taylor_lapl_matrix = []
	for i in range(0, n):
		for j in range(0, n):
			M[i, j] = fe.integrate(point, f[i]*f[j])*dV		
			L[i, j] = fe.integrate(point, inner(fe.symbol_fff() * g[i], g[j]))

			taylor_mass_matrix.append(ast.Assignment(sp.symbols(f"element_matrix[{i*n+j}]"), M[i, j]))	
			taylor_lapl_matrix.append(ast.Assignment(sp.symbols(f"element_matrix[{i*n+j}]"), L[i, j]))	

	stop = perf_counter()
	console.print(f'Taylor) Elapsed  {stop - start} seconds')

	# Taylor expansion
	print("//Taylor expansion")
	# print("//Mass")
	# c_code(taylor_mass_matrix)

	print("//Laplacian")
	c_code(taylor_lapl_matrix)

	u = coeffs('u', n)
	values = coeffs('element_vector', n)

	Lu = L * u

	taylor_lapl_apply = []
	for i in range(0, n):
		taylor_lapl_apply.append(ast.Assignment(values[i], Lu[i]))

	# print("//Laplacian apply")
	# c_code(taylor_lapl_apply)

	print("//------------------------------------------------")

	start = perf_counter()
	print("//Symbolic integation")
	f = Field(fe, coeffs('u', fe.n_nodes()))
	op = MassOp(f, fe, True)
	sym_mass_matrix = op.matrix()

	# print("//Mass")
	# c_code(sym_mass_matrix)
	
	sym_lapl_matrix = LaplaceOp(fe, True).hessian()

	print("//Laplacian")
	c_code(sym_lapl_matrix)

	stop = perf_counter()
	console.print(f'Symbolic) Elapsed  {stop - start} seconds')


	# TESTING

	if False:
		# Verification
		if dim == 2:
			spx = coeffs('px', n)
			spy = coeffs('py', n)
			px = sp.Matrix(n, 1, [0, 1, 1, 0])
			py = sp.Matrix(n, 1, [0, 0, 2, 2])
		else:
			spx = coeffs('x', n)
			spy = coeffs('y', n)
			spz = coeffs('z', n)
			px = sp.Matrix(n, 1, [0, 1, 1, 0, 0, 1, 1, 0])
			py = sp.Matrix(n, 1, [0, 0, 1, 1, 0, 0, 1, 1])
			pz = sp.Matrix(n, 1, [0, 0, 0, 0, 1, 1, 1, 1])


		print("//Oracle (symbolic)")
		eval_expr = []
		for e in sym_mass_matrix:
			eval_expr.append(subsmat(subsmat(e, spx, px), spy, py))
		c_code(eval_expr)

		print("//Actual (taylor)")
		eval_expr = []
		for e in taylor_mass_matrix:
			se = subsmat(subsmat(e, spx, px), spy, py)

			if dim == 3:
				se = subsmat(se, spz, pz),
				eval_expr.append(se[0])
			else:
				eval_expr.append(se)	
		c_code(eval_expr)


