#!/usr/bin/env python3

from tet10 import *
from tet20 import *
from tet4 import *
from tri6 import *
from quad4 import *
from mass_op import *
from laplace_op import *
from stensor import Tensor3

sp.init_printing()

def try_laplacian(fe):
	dim = fe.spatial_dim()

	start = perf_counter()

	c = fe.barycenter()

	order = -1

	point = coeffs('p', dim)
	f = fe.taylor_expand_fun(c, point, order)
	g = fe.taylor_expand_grad(c, point, order)

	T2 = fe.eval_hessian(point)
	T2_nnz = sp.zeros(dim, dim)

	for t2 in T2:
		for d1 in range(0, dim):
			for d2 in range(0, dim):
				T2_nnz[d1, d2] += (t2[d1, d2] != 0) * 1

	sp.pprint(T2_nnz)

	T3 = fe.eval_diff3(point)
	T3_nnz = Tensor3(dim, dim, dim)
	for t3 in T3:
		T3_nnz.iadd(t3.nnz_op())
	
	sp.pprint(T3_nnz.d)
	# exit(0)

	f0 = fe.fun(c)
	g0 = fe.eval_grad(c)

	n = fe.n_nodes()
	M = sp.zeros(n, n)
	L = sp.zeros(n, n)
	q = fe.quadrature_point()
	# dV = subsmat(fe.measure(q), q, c)
	dV = fe.symbol_jacobian_determinant()

	Ldiff = sp.zeros(n, n)
	L0 = sp.zeros(n, n)

	taylor_mass_matrix = []
	taylor_lapl_matrix = []
	
	for i in range(0, n):
		for j in range(0, n):
			M[i, j] = fe.integrate(point, f[i]*f[j])*dV		
			L[i, j] = fe.integrate(point, inner(fe.symbol_fff() * g[i], g[j]))
			L0[i, j] = inner(fe.symbol_fff() * g0[i], g0[j])
			Ldiff[i, j] = L[i, j] - L0[i, j]
			Ldiff[i, j] = sp.simplify(Ldiff[i, j])

			taylor_mass_matrix.append(ast.Assignment(sp.symbols(f"element_matrix[{i*n+j}]"), M[i, j]))	
			taylor_lapl_matrix.append(ast.Assignment(sp.symbols(f"element_matrix[{i*n+j}]"), L[i, j]))	
			
	stop = perf_counter()
	console.print(f'Taylor) Elapsed  {stop - start} seconds')

	# Taylor expansion
	print("//Taylor expansion")
	print("//Mass")
	c_code(taylor_mass_matrix)

	# print("//Laplacian")
	# c_code(taylor_lapl_matrix)

	

	u = coeffs('u', n)
	values = coeffs('element_vector', n)

	Lu = L * u
	Lu_diff = Ldiff * u

	taylor_lapl_apply = []
	taylor_lapl_diff = []
	for i in range(0, n):
		taylor_lapl_apply.append(ast.Assignment(values[i], Lu[i]))
		taylor_lapl_diff.append(ast.Assignment(values[i], Lu_diff[i]))	

	# print("//Laplacian apply")
	# c_code(taylor_lapl_apply)

	# print("//Laplacian apply diff")
	# c_code(taylor_lapl_diff)

	print("//------------------------------------------------")

	start = perf_counter()
	print("//Symbolic integation")
	f = Field(fe, coeffs('u', fe.n_nodes()))
	op = MassOp(f, fe, True)
	sym_mass_matrix = op.matrix()

	print("//Mass")
	c_code(sym_mass_matrix)
	
	sym_lapl_matrix = LaplaceOp(fe, True).hessian()

	# print("//Laplacian")
	# c_code(sym_lapl_matrix)

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

def try_laplacian_symbolic(fe):
	dim = fe.spatial_dim()
	point = coeffs('p', dim)
	c = fe.barycenter()

	order = -1

	trial = "trial"
	test = "test"
	gi = fe.taylor_grad_symbolic(trial, c, point, order)
	gj = fe.taylor_grad_symbolic(test, c, point, order)

	fff_x_gi = fe.symbol_fff() * gi
	Lij = 0

	print("// Integrating...")
	for d in range(0, dim):
		print(f"// {d})")
		Lij += fe.integrate(point, fff_x_gi[d] * gj[d])

	var = sp.symbols(f"Lij")
	expr = [ast.Assignment(var, sp.simplify(Lij))]
	c_code(expr)

	trial_g_nnz = fe.grad_nnz(trial)
	trial_H_nnz = fe.hessian_nnz(trial)
	trial_T3_nnz = fe.diff3_nnz(trial)


	g = fe.eval_grad(c) 
	H = fe.eval_hessian(c)
	T3 = fe.eval_diff3(c)

	u = coeffs('u', fe.n_nodes())

	gu = 0
	Hu = 0
	T3u = 0
	first = True

	for i in range(0, fe.n_nodes()):
		fe_vals = []
		print(f"// Fun {i})")

		if first:
			gu = u[i] * g[i]
			Hu = u[i] * compress_nnz(trial_H_nnz, H[i])
			T3u = u[i] * trial_T3_nnz.compress_nnz(T3[i])
			first = False
		else:
			gu += u[i] * g[i]
			Hu += u[i] * compress_nnz(trial_H_nnz, H[i])
			T3u += u[i] * trial_T3_nnz.compress_nnz(T3[i])

		fe_vals.extend(assign_nnz_matrix(trial_g_nnz, g[i]))
		fe_vals.extend(assign_nnz_matrix(trial_H_nnz, H[i]))
		fe_vals.extend(trial_T3_nnz.assign_nnz(T3[i]))
		c_code(fe_vals)
		print('//---------------------------------')

	print("// Evals")
	evals_expr = []
	evals_expr.extend(assign_matrix("gu", gu))
	evals_expr.extend(assign_matrix("Hu", Hu))
	evals_expr.extend(assign_matrix("diff3u", T3u))
	c_code(evals_expr)


if __name__ == '__main__':
	fe = Hex8(False)
	# fe = Tet10()
	try_laplacian_symbolic(fe)

	