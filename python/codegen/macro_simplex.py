#!/usr/bin/env python3

import numpy as np
# import pdb
from fe import FE
import sympy as sp
from sfem_codegen import *

dim = 3

if dim == 2:
	sub_simplices = np.array([
		[0, 3, 5],
		[3, 1, 4],
		[5, 4, 2],
		[3, 4, 5]
	])

	edges = [
		[0, 1],
		[1, 2],
		[2, 0],
	]

	unique = [0, 3]

	c = coeffs("fff", 3)
	FFFs = sp.Matrix(dim, dim, [
		c[0], c[1],
		c[1], c[2]
	])

else:
	sub_simplices = [
	[0, 4, 6, 7],
	[4, 1, 5, 8],
	[6, 5, 2, 9],
	[7, 8, 9, 3],
	[4, 5, 6, 8],
	[7, 4, 6, 8],
	[6, 5, 9, 8],
	[7, 6, 9, 8]]

	edges = [
		[0, 1],
		[1, 2],
		[0, 2],
		[0, 3],
		[1, 3],
		[2, 3]
	]

	unique = [0, 4, 5, 6, 7]

	c = coeffs("fff", 6)
	FFFs = sp.Matrix(dim, dim, [
		c[0], c[1], c[2], 
		c[1], c[3], c[4],
		c[2], c[4], c[5]
	])

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

def subJ_generic(micro_ref, FFFs):
	Am = sp.zeros(dim, dim)

	for d1 in range(0, dim):
		for d2 in range(0, dim):
			Am[d2, d1] = micro_ref[d1+1][d2] - micro_ref[0][d2]
	
	detAm = determinant(Am)
	Aminv = inverse(Am)

	# inv(J * A) = (inv(A) * inv(J))
	# inv(J * A)^T = (inv(J)^T * inv(A)^T)
	# <(inv(J)^T * inv(A)^T) gi, (inv(J)^T * inv(A)^T) gj> 
	# <(inv(J)^T * inv(A)*T)^T * inv(J)^T * inv(A)^T gi, gj>
	# <inv(A) * inv(J) * inv(J)^T * inv(A)^T gi, gj>
	# <inv(A) * FFF * inv(A)^T gi, gj>


	FFAms = Aminv * FFFs * Aminv.T * detAm

	print("------------------")
	print(Aminv)
	print(detAm)
	print(FFAms)
	return FFAms

def subJ(micro_ref):
	for d1 in range(0, dim):
		for d2 in range(d1+1, dim):
			FFFs[d2, d1] = FFFs[d1, d2] 

	return subJ_generic(micro_ref, FFFs)

def fff_code_basic(FFF):
	funs = []
	tpl = read_file('tpl/macro_sub_jacobian_tpl.c')
	code0to2 = tpl.format(
		NUM="0to2",
		CODE=c_gen(assign_fff("sub_fff", FFF[0]))
	)

	code3 = tpl.format(
		NUM="3",
		CODE=c_gen(assign_fff("sub_fff", FFF[3]))
	)

	funs = [code0to2, code3]
	return funs

def fff_code(name, FFF):
	funs = []
	tpl = read_file('tpl/macro_sub_jacobian_tpl.c')
	code = tpl.format(
		NUM=name,
		CODE=c_gen(assign_fff("sub_fff", FFF))
	)

	return code

class MacroSimplex:
	def __init__(self):
		points = []
		points.append(sp.zeros(dim, 1))
		for d1 in range(0, dim):
			p = sp.zeros(dim, 1)
			p[d1] = 1
			points.append(p)

		for e in edges:
			p = (points[e[0]] + points[e[1]])/2
			points.append(p)

		self.points = np.array(points)

	def fff_level_n(self, n_levels):
		levels = [[]] * n_levels
		x = self.points

		for ss in sub_simplices:
			refpattern = x[ss]
			fff = subJ(refpattern)
			levels[0].append(fff)

		for l in range(1, n_levels):
			n_ffs = len(levels[l-1])
			levels[l] = [0]*(n_ffs * 2)

			for i in range(0, n_ffs):
				fff = levels[l-1][i]
				for ss in sub_simplices:
					refpattern = x[ss]
					sub_fff = subJ_generic(refpattern, fff)
					levels[l].append(sub_fff)
		return levels

nl = 1
fffl = MacroSimplex().fff_level_n(nl)

for l in range(0, nl):
	num = 0
	print(f'level {l} #fff {len(fffl[l])}')
	# for i in unique:
	for i in range(0, len(fffl[l])):
		f = fffl[l][i]
		print(fff_code(f'{l}_{num}',f))
		num += 1
