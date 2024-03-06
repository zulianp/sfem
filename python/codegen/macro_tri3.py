#!/usr/bin/env python3

import pdb
# from sfem_codegen import *

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
	x0m = micro_ref[0]
	x1m = micro_ref[1]
	x2m = micro_ref[2]

	um = x1m - x0m
	vm = x2m - x0m
	Am = sp.Matrix(2, 2, [um[0], vm[0], um[1], vm[1]])
	Aminv = inv2(Am)
	Rm = Am
	Rminv = inv2(Rm)
	FFAms = Rm * FFFs * Rm.T
	return FFAms

def subJ(micro_ref):
	FFFs = matrix_coeff("fff", 2, 2)
	FFFs[1, 0] = FFFs[0, 1] 
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

class MacroTri3:
	def __init__(self):
		self.x0 = vec2(0., 0.)
		self.x1 = vec2(1., 0.)
		self.x2 = vec2(0., 1.)
		self.x3 = (self.x0 + self.x1)/2
		self.x4 = (self.x1 + self.x2)/2
		self.x5 = (self.x2 + self.x0)/2

	def fff_level_1(self):
		x0 = self.x0 
		x1 = self.x1 
		x2 = self.x2 
		x3 = self.x3 
		x4 = self.x4 
		x5 = self.x5 

		FFFA0toA2 = subJ([x0, x3, x5])
		FFF = [
			FFFA0toA2,
			FFFA0toA2,
			FFFA0toA2,
			subJ([x3, x4, x5])
		]

		return fff_code_basic(FFF)
	
	def fff_level_n(self, n_levels):
		x0 = self.x0 
		x1 = self.x1 
		x2 = self.x2 
		x3 = self.x3 
		x4 = self.x4 
		x5 = self.x5 

		levels = [[]] * n_levels
		FFFA0toA2 = subJ([x0, x3, x5])
		FFFA3 = subJ([x3, x4, x5])
		levels[0].append(FFFA0toA2)
		levels[0].append(FFFA3)

		for l in range(1, n_levels):
			n_ffs = len(levels[l-1])
			print(n_ffs)

			levels[l] = [0]*(n_ffs * 2)

			for i in range(0, n_ffs):
				fff = levels[l-1][i]
				fffa = subJ_generic([x0, x3, x5], fff)
				fffb = subJ_generic([x3, x4, x5], fff)
				levels[l][i*2]= fffa
				levels[l][i*2+1] = fffb
		return levels




# fff = MacroTri3().fff_level_1()
# for f in fff:
# 	print(f)

nl = 1
fffl = MacroTri3().fff_level_n(nl)

for l in range(0, nl):
	num = 0
	print(f'level {l} #fff {len(fffl[l])}')
	for f in fffl[l]:
		print(fff_code(f'{l}_{num}',f))
		num += 1

