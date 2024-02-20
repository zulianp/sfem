#!/usr/bin/env python3

import numpy as np
import pdb

dim = 2

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

def assign_matrix(name, mat):
	rows, cols = mat.shape

	expr = []
	for i in range(0, rows):
		for j in range(0, cols):
			var = sp.symbols(f'{name}[{i*cols + j}]')
			expr.append(ast.Assignment(var, mat[i, j]))
	return expr

from fe import FE
import sympy as sp
from sfem_codegen import *

def subJ_generic(micro_ref, FFFs):
	Am = sp.zeros(dim, dim)

	for d1 in range(0, dim):
		for d2 in range(0, dim):
			Am[d1, d2] = micro_ref[d1+1, d2] - micro_ref[0, d2]
	
	Aminv = inv2(Am)
	Rm = Am
	Rminv = inv2(Rm)
	FFAms = Rm * FFFs * Rm.T
	return FFAms

def subJ(micro_ref):
	FFFs = matrix_coeff("fff", dim, dim)
	for d1 in range(0, dim):
		for d2 in range(d1+1, dim):
			FFFs[d2, d1] = FFFs[d1, d2] 

	return subJ_generic(micro_ref, FFFs)

def fff_code_basic(FFF):
	funs = []
	tpl = read_file('tpl/macro_sub_jacobian_tpl.c')
	code0to2 = tpl.format(
		NUM="0to2",
		CODE=c_gen(assign_matrix("sub_fff", FFF[0]))
	)

	code3 = tpl.format(
		NUM="3",
		CODE=c_gen(assign_matrix("sub_fff", FFF[3]))
	)

	funs = [code0to2, code3]
	return funs

def fff_code(name, FFF):
	funs = []
	tpl = read_file('tpl/macro_sub_jacobian_tpl.c')
	code = tpl.format(
		NUM=name,
		CODE=c_gen(assign_matrix("sub_fff", FFF))
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
	for i in unique:
		f = fffl[l][i]
		print(fff_code(f'{l}_{num}',f))
		num += 1
