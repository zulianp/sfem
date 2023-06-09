import numpy as np
import sympy as sp
from sympy.utilities.codegen import codegen
import sympy.codegen.ast as ast
import rich
from time import perf_counter

from rich.syntax import Syntax
console = rich.get_console()

def c_log(expr):
	console.print(expr)


# from sympy.matrices.dense import eye
# from sympy.polys.matrices import DomainMatrix
# from sympy.physics.quantum import TensorProduct
# from sympy.physics.matrices import msigma

def det2(mat):
    return mat[0,0] * mat[1,1] - mat[1,0] * mat[0,1];

def inv2(mat):
	mat_inv = sp.zeros(2, 2)
	det = det2(mat)

	mat_inv[0] = mat[1,1] / det
	mat_inv[1] = -mat[0,1] / det
	mat_inv[2] = -mat[1,0] / det
	mat_inv[3] = mat[0,0] / det
	return mat_inv

def det3(mat):
    return mat[0, 0] * mat[1, 1] * mat[2, 2] + mat[0, 1] * mat[1, 2] * mat[2, 0] + mat[0, 2] * mat[1, 0] * mat[2, 1] - mat[0, 0] * mat[1, 2] * mat[2, 1] - mat[0, 1] * mat[1, 0] * mat[2, 2] - mat[0, 2] * mat[1, 1] * mat[2, 0]

def inv3(mat):
	# Sympy version (same but slower)
	# return mat.inv()
    mat_inv = sp.zeros(3, 3)
    det = det3(mat)
    mat_inv[0, 0] = (mat[1, 1] * mat[2, 2] - mat[1, 2] * mat[2, 1]) / det
    mat_inv[0, 1] = (mat[0, 2] * mat[2, 1] - mat[0, 1] * mat[2, 2]) / det
    mat_inv[0, 2] = (mat[0, 1] * mat[1, 2] - mat[0, 2] * mat[1, 1]) / det
    mat_inv[1, 0] = (mat[1, 2] * mat[2, 0] - mat[1, 0] * mat[2, 2]) / det
    mat_inv[1, 1] = (mat[0, 0] * mat[2, 2] - mat[0, 2] * mat[2, 0]) / det
    mat_inv[1, 2] = (mat[0, 2] * mat[1, 0] - mat[0, 0] * mat[1, 2]) / det
    mat_inv[2, 0] = (mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0]) / det
    mat_inv[2, 1] = (mat[0, 1] * mat[2, 0] - mat[0, 0] * mat[2, 1]) / det
    mat_inv[2, 2] = (mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]) / det
    return mat_inv

def c_gen(expr, dump=False):
    console.print("--------------------------")
    console.print(f'Running cse')
    start = perf_counter()

    sub_expr, simpl_expr = sp.cse(expr)

    # sub_ops = sp.count_ops(sub_expr, visual=True)
    # result_ops = sp.count_ops(simpl_expr, visual=True)
    # cost = f'FLOATING POINT OPS!\n//\t- Result: {result_ops}\n//\t- Subexpressions: {sub_ops}'
    
    printer = sp.printing.c.C99CodePrinter()
    lines = []

    for var,expr in sub_expr:
        lines.append(f'const real_t {var} = {printer.doprint(expr)};')

    for v in simpl_expr:
        lines.append(printer.doprint(v))

    code_string=f'\n'.join(lines)

    stop = perf_counter()
    console.print(f'Elapsed  {stop - start} seconds')
    console.print("--------------------------")
    console.print(f'generated code')

    # code_string = f'//{cost}\n' + code_string
    code_string = f'//TODO COST\n' + code_string

    if dump:
        console.print(code_string)

    return code_string

def c_code(expr):
	code_string = c_gen(expr)
	console.print(code_string)

def inner(l, r):
	rows, cols = l.shape
	
	ret = 0
	for d1 in range(0, rows):
		for d2 in range(0, cols):
			ret += l[d1, d2] * r[d1, d2]

	return ret

def dot3(l, r):
	ret = 0
	for d1 in range(0, 3):
		ret += l[d1] * r[d1]

	return ret

def tr(mat):
	rows, cols = mat.shape
	assert(rows == cols)

	ret = 0
	for d1 in range(0, rows):
		ret += mat[d1, d1]
	return ret

def ref_fun(x, y, z):
	return [
	 1 - x - y - z, 
	 x,
	 y,
	 z
	]

def fun(x, y, z):
	xmb = x - b[0]
	ymb = y - b[1]
	zmb = z - b[2]

	xref = Ainv[0, 0] * xmb + Ainv[0, 1] * ymb  + Ainv[0, 2] * zmb
	yref = Ainv[1, 0] * xmb + Ainv[1, 1] * ymb  + Ainv[1, 2] * zmb
	zref = Ainv[2, 0] * xmb + Ainv[2, 1] * ymb  + Ainv[2, 2] * zmb
	return ref_fun(xref, yref, zref)

qx, qy, qz, qw = sp.symbols('qx qy qz qw', real=True)

# Element coordinates
# x0, x1, x2, x3 = sp.symbols('x0 x1 x2 x3', real=True)
# y0, y1, y2, y3 = sp.symbols('y0 y1 y2 y3', real=True)
# z0, z1, z2, z3 = sp.symbols('z0 z1 z2 z3', real=True)

x0, x1, x2, x3 = sp.symbols('px0 px1 px2 px3', real=True)
y0, y1, y2, y3 = sp.symbols('py0 py1 py2 py3', real=True)
z0, z1, z2, z3 = sp.symbols('pz0 pz1 pz2 pz3', real=True)

# Quadrature points (Physical coordinates)
q = sp.Matrix(3, 1, [qx, qy, qz])

# Affine transformation
A = sp.Matrix(3, 3, [
	 x1 - x0, x2 - x0, x3 - x0,
	 y1 - y0, y2 - y0, y3 - y0,
	 z1 - z0, z2 - z0, z3 - z0,
	])

Ainv = inv3(A)

b = sp.Matrix(3, 1, [x0, y0, z0])

def symm_grad(x, y, z):
	ret = []
	f = fun(x, y, z)

	i = 0
	for fi in f:
		gix = sp.simplify(sp.diff(fi, x))
		giy = sp.simplify(sp.diff(fi, y))
		giz = sp.simplify(sp.diff(fi, z))
		g = [gix, giy, giz]

		for d1 in range(0, 3):
			eps = sp.Matrix(3, 3, [0, 0, 0, 
								   0, 0, 0, 
								   0, 0, 0])

			for d2 in range(0, 3):
				eps[d1, d2] = g[d2]

			simmetrized_eps = (eps + eps.T) / 2
			ret.append(simmetrized_eps)

		i += 1
	return ret

def tgrad(x, y, z):
	ret = []
	f = fun(x, y, z)

	i = 0
	for fi in f:
		gix = sp.simplify(sp.diff(fi, x))
		giy = sp.simplify(sp.diff(fi, y))
		giz = sp.simplify(sp.diff(fi, z))
		g = [gix, giy, giz]

		for d1 in range(0, 3):
			G = sp.Matrix(3, 3, [0, 0, 0, 
								   0, 0, 0, 
								   0, 0, 0])

			for d2 in range(0, 3):
				G[d1, d2] = g[d2]

			ret.append(G)

		i += 1
	return ret

def generic_grad(prefix):
	gx, gy, gz = sp.symbols(f'{prefix}[0] {prefix}[1] {prefix}[2]')
	g = sp.Matrix(3, 1, [gx, gy, gz])
	return g

def tensorize_grad(g):
	ret = []

	for d1 in range(0, 3):
		G = sp.Matrix(3, 3, [0, 0, 0, 
							 0, 0, 0, 
							 0, 0, 0])

		for d2 in range(0, 3):
			G[d1, d2] = g[d2]

		ret.append(G)
	return ret

def generic_symm_grad(prefix):
	ret = []

	g = generic_grad(prefix)

	for d1 in range(0, 3):
		G = sp.Matrix(3, 3, [0, 0, 0, 
							 0, 0, 0, 
							 0, 0, 0])

		for d2 in range(0, 3):
			G[d1, d2] = g[d2]

		G = (G + G.T)/2

		ret.append(G)
	return ret

def subsmat(expr, oldmat, newmat):
	rows, cols = oldmat.shape
	nrows, ncols = newmat.shape

	assert rows == nrows
	assert cols == ncols

	for d1 in range(0, rows):
		for d2 in range(0, cols):
			expr = expr.subs(oldmat[d1, d2], newmat[d1, d2])
	
	return expr

def subsmat3x3(expr, oldmat, newmat):
	for d1 in range(0, 3):
		for d2 in range(0, 3):
			expr = expr.subs(oldmat[d1, d2], newmat[d1, d2])
	return expr

def subsvec3(expr, oldvec, newvec):
	for d1 in range(0, 3):
		expr = expr.subs(oldvec[d1], newvec[d1])
	return expr

def coeffs(name, n):
	list_coeffs = []

	for i in range(0, n):
		ui= sp.symbols(f'{name}[{i}]', real=True)
		list_coeffs.append(ui)

	ret = sp.Matrix(n, 1, list_coeffs)
	return ret


def norm2(v):
	ret = 0.0;

	rows, cols = v.shape

	for d1 in range(0, rows):
		for d2 in range(0, cols):
			ret += v[d1, d2] * v[d1, d2]

	return sp.sqrt(ret)

def cross(a, b):
	return sp.Matrix(3, 1, [
		a[1] * b[2] - a[2]*b[1],
		a[2] * b[0] - a[0]*b[2],
		a[0] * b[1] - a[1]*b[0]
	])


def stot(x):
	return sp.Matrix(1, 1, [x])

def vec3(x, y, z):
	return sp.Matrix(3, 1, [x, y, z])

def eigenvalues(A):
	s0, s1, s2 = sp.symbols('s0 s1 s2', real=True)
	s3, s4, s5 = sp.symbols('s3 s4 s5', real=True)
	s6, s7, s8 = sp.symbols('s6 s7 s8', real=True)

	S = sp.Matrix(3, 3, [s0, s1, s2,
						 s3, s4, s5,
						 s6, s7, s8])
	Se = S.eigenvals()
	ret = sp.Matrix(3, 1, [0]*3)

	d = 0
	for e in Se:
		e_subs = e	
		for i in range(0, 3):
			for j in range(0, 3):
				e_subs = e_subs.subs(S[i, j], A[i, j])
		ret[d] = e_subs
		d += 1

	return ret

